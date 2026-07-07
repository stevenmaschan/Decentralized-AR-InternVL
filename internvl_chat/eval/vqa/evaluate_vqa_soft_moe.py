"""Soft mixture-of-experts evaluation for VQA-style benchmarks (default: TextVQA).

Unlike the hard top-1 pipeline (``evaluate_vqa.py`` run separately on each
``clusterN.jsonl`` split), this script keeps *all* experts live for every
sample. A CLIP-ViT-B/16 single-stage router turns each image into a soft weight
vector ``(w_0, ..., w_{K-1})`` over the experts, and the experts are combined at
*every* autoregressive decoding step by mixing their next-token distributions:

    p = sum_m w_m * softmax(z_m)          # 'prob' combine (default)
    p = softmax(sum_m w_m * z_m)          # 'logit' combine

The router weights are fixed for the whole generation of a sample (the router
sees the image, not the tokens). Decoding is greedy (temperature 0), matching the
default of the hard evaluation, so results are directly comparable.

Assumptions:
  * K experts, each a full InternVL2.5-1B fine-tune sharing an identical
    tokenizer / vocabulary (so their logits are aligned and mixable).
  * Expert m is placed on ``cuda:m`` (model parallel, not data parallel). With
    K=2 that is expert0 -> cuda:0, expert1 -> cuda:1. Logits are combined on
    cuda:0.
  * The single-stage router is already trained: ``clustering_centroids.npy``
    lives in ``--clustering-results-dir``.

Example (K=2, CLIP-ViT-B/16 single-stage):

    python eval/vqa/evaluate_vqa_soft_moe.py \
        --datasets textvqa_val \
        --expert-checkpoints work_dirs/cluster0,work_dirs/cluster1 \
        --clustering-results-dir clustering/balanced-kmeans_vit-base-patch-16_2-coarse \
        --clip-model openai/clip-vit-base-patch16 \
        --router-temperature 1.0 \
        --combine prob \
        --out-dir results_soft_moe
"""

import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from internvl.conversation import get_conv_template
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

# Reuse the exact dataset / prompts / post-processing / metric of the hard path
# so soft-MoE numbers are directly comparable.
from evaluate_vqa import VQADataset, collate_fn, ds_collections, post_process
from textvqa_eval import TextVQAAccuracyEvaluator

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


# ---------------------------------------------------------------------------
# Router: CLIP-ViT-B/16 single-stage balanced k-means -> soft weights
# ---------------------------------------------------------------------------
class SoftRouter:
    """Turns an image into soft expert weights using single-stage centroids.

    Mirrors the feature extraction / normalization of ``clustering/split_dataset.py``
    (single-stage branch), but returns ``softmax(cos_sim / tau)`` instead of an
    argmax. tau -> 0 recovers hard top-1 routing (useful as a sanity check).
    """

    def __init__(self, clustering_results_dir, clip_model_name, temperature,
                 device, prefix='clustering'):
        centroids_file = os.path.join(clustering_results_dir, f'{prefix}_centroids.npy')
        if not os.path.exists(centroids_file):
            raise FileNotFoundError(
                f'Single-stage centroids not found: {centroids_file}. '
                f'This script only supports single-stage routing.')
        centroids = np.load(centroids_file)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        self.centroids = torch.from_numpy(centroids).float().to(device)  # (K, D)
        self.n_experts = self.centroids.shape[0]
        self.temperature = float(temperature)
        self.device = device

        print(f'Loading CLIP router: {clip_model_name} on {device} '
              f'(K={self.n_experts}, tau={self.temperature})', flush=True)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    @torch.no_grad()
    def weights(self, image_path):
        """Return expert weights (torch tensor, shape (K,)) for one image."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors='pt').to(self.device)
        feat = self.clip_model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=1, keepdim=True)          # (1, D), normalized
        sims = (self.centroids @ feat.squeeze(0)).float()      # (K,) cosine sims
        return F.softmax(sims / self.temperature, dim=0)       # (K,)


# ---------------------------------------------------------------------------
# Expert wrapper: build spliced input embeddings + step the LLM with a cache
# ---------------------------------------------------------------------------
class Expert:
    """One InternVL expert pinned to a single device, driven token-by-token."""

    def __init__(self, checkpoint, device, img_context_token_id):
        self.device = device
        model = InternVLChatModel.from_pretrained(
            checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
        self.model = model.to(device)
        self.model.system_message = ''
        self.model.img_context_token_id = img_context_token_id
        self.lm = self.model.language_model
        self.embed = self.lm.get_input_embeddings()

    @torch.no_grad()
    def build_prefill_embeds(self, input_ids, pixel_values):
        """Splice ViT features into the token embeddings (see model.generate)."""
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        vit_embeds = self.model.extract_feature(pixel_values)
        embeds = self.embed(input_ids)
        B, N, C = embeds.shape
        embeds = embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == self.model.img_context_token_id)
        assert selected.sum() != 0, 'no <IMG_CONTEXT> tokens found in prompt'
        embeds[selected] = vit_embeds.reshape(-1, C).to(embeds.dtype)
        return embeds.reshape(B, N, C)

    @torch.no_grad()
    def prefill(self, inputs_embeds, seq_len):
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        self.past = out.past_key_values
        self.attention_mask = attention_mask
        return out.logits[:, -1, :]  # (1, V)

    @torch.no_grad()
    def step(self, next_token):
        """Advance one token (id chosen from the *combined* distribution)."""
        next_token = next_token.to(self.device).reshape(1, 1)
        inputs_embeds = self.embed(next_token)
        past_len = self.attention_mask.shape[1]
        self.attention_mask = torch.cat(
            [self.attention_mask,
             torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1)
        position_ids = torch.tensor([[past_len]], device=self.device)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=self.attention_mask,
            position_ids=position_ids,
            past_key_values=self.past,
            use_cache=True,
        )
        self.past = out.past_key_values
        return out.logits[:, -1, :]  # (1, V)


# ---------------------------------------------------------------------------
# Lockstep soft-MoE greedy decoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def soft_moe_generate(experts, tokenizer, template, input_ids, pixel_values,
                      weights, max_new_tokens, eos_token_id, combine):
    """Greedy-decode all experts in lockstep, mixing their distributions.

    weights: tensor (K,) on the combine device (experts[0].device).
    Returns the list of generated token ids (excludes the prompt).
    """
    combine_device = experts[0].device
    weights = weights.to(combine_device)

    # Prefill each expert with its own spliced embeddings.
    logits = []
    for m, expert in enumerate(experts):
        embeds = expert.build_prefill_embeds(input_ids, pixel_values)
        logits.append(expert.prefill(embeds, embeds.shape[1]))

    generated = []
    for _ in range(max_new_tokens):
        next_token = mix_and_pick(logits, weights, combine_device, combine)
        token_id = int(next_token.item())
        if token_id == eos_token_id:
            break
        generated.append(token_id)
        # Feed the shared next token back into every expert.
        logits = [expert.step(next_token) for expert in experts]
    return generated


def mix_and_pick(logits, weights, device, combine):
    """Combine per-expert last-position logits into one next-token id (greedy)."""
    logits = [z.to(device).float() for z in logits]
    if combine == 'prob':
        # Mixture of distributions: p = sum_m w_m * softmax(z_m)
        probs = sum(w * F.softmax(z, dim=-1) for w, z in zip(weights, logits))
        return probs.argmax(dim=-1)
    elif combine == 'logit':
        # Geometric / product mixture: z = sum_m w_m * z_m
        z = sum(w * zz for w, zz in zip(weights, logits))
        return z.argmax(dim=-1)
    raise ValueError(f'unknown combine mode: {combine}')


def build_prompt(model, tokenizer, template_name, system_message, question,
                 num_patches, num_image_token):
    """Reproduce model.chat() prompt construction (single image, single turn)."""
    if '<image>' not in question:
        question = '<image>\n' + question
    template = get_conv_template(template_name)
    template.system_message = system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors='pt')
    return model_inputs['input_ids'], template


def evaluate(args):
    # ------------------------------------------------------------------ setup
    checkpoints = [c.strip() for c in args.expert_checkpoints.split(',') if c.strip()]
    n_experts = len(checkpoints)
    assert n_experts >= 2, 'need at least 2 expert checkpoints'
    if torch.cuda.device_count() < n_experts:
        print(f'[warn] {n_experts} experts but only {torch.cuda.device_count()} '
              f'GPUs visible; experts will share devices (may OOM).')
    devices = [f'cuda:{min(m, torch.cuda.device_count() - 1)}' for m in range(n_experts)]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoints[0], trust_remote_code=True, use_fast=False)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    experts = [Expert(ckpt, dev, img_context_token_id)
               for ckpt, dev in zip(checkpoints, devices)]
    cfg = experts[0].model.config
    image_size = cfg.force_image_size or cfg.vision_config.image_size
    use_thumbnail = cfg.use_thumbnail
    num_image_token = experts[0].model.num_image_token
    template_name = cfg.template
    # EOS is the conversation separator token, exactly as in model.chat().
    sep = get_conv_template(template_name).sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(sep)

    router = SoftRouter(args.clustering_results_dir, args.clip_model,
                        args.router_temperature, experts[0].device)
    assert router.n_experts == n_experts, (
        f'router has {router.n_experts} centroids but {n_experts} experts given')

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------- per dataset
    for ds_name in args.datasets:
        base_prompt = 'Answer the question using a single word or phrase.'
        input_prompt = '' if 'ai2d' in ds_name else base_prompt
        max_new_tokens = args.max_new_tokens or ds_collections[ds_name]['max_new_tokens']

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=0,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
        )
        # Access raw records too, so the router can read each image path.
        records = [json.loads(l.strip()) for l in dataset.test]

        outputs = []
        for idx in tqdm(range(len(dataset)), desc=ds_name):
            item = dataset[idx]
            pixel_values = item['pixel_values']            # (num_patches, 3, H, W)
            num_patches = pixel_values.shape[0]
            question = item['question']
            question_id = item['question_id']

            weights = router.weights(records[idx]['image'])  # (K,)

            input_ids, template = build_prompt(
                experts[0].model, tokenizer, template_name,
                experts[0].model.system_message, question,
                num_patches, num_image_token)

            gen_ids = soft_moe_generate(
                experts, tokenizer, template, input_ids, pixel_values,
                weights, max_new_tokens, eos_token_id, args.combine)
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
            answer = answer.split(sep)[0].strip()

            outputs.append({
                'question': question,
                'question_id': question_id,
                'answer': answer,
                'router_weights': [round(float(w), 4) for w in weights.tolist()],
            })

        # ------------------------------------------------------------- scoring
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(
            args.out_dir, f'{ds_name}_soft_moe_{args.combine}_tau{args.router_temperature}_{time_prefix}.json')
        json.dump(outputs, open(results_file, 'w'))
        print(f'Results saved to {results_file}')

        metric = ds_collections[ds_name]['metric']
        if metric == 'vqa_score':
            evaluator = TextVQAAccuracyEvaluator()
            annotation = json.load(open(ds_collections[ds_name]['annotation']))['annotations']
            qid2answers = {a['question_id']: [x['answer'] for x in a['answers']]
                           for a in annotation}
            for it in outputs:
                it['pred_answer'] = it['answer']
                it['gt_answers'] = qid2answers[it['question_id']]
            accuracy = evaluator.eval_pred_list(outputs)
            print(f'{ds_name}  soft-MoE ({args.combine}, tau={args.router_temperature})  '
                  f'accuracy = {accuracy}')
        else:
            print(f'[warn] metric "{metric}" not wired up in soft-MoE script; '
                  f'predictions saved to {results_file} for external scoring.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='textvqa_val',
                        help='comma-separated ds_collections keys (full, unsplit sets)')
    parser.add_argument('--expert-checkpoints', type=str, required=True,
                        help='comma-separated expert checkpoints, ordered by cluster id')
    parser.add_argument('--clustering-results-dir', type=str, required=True,
                        help='dir with single-stage clustering_centroids.npy')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--router-temperature', type=float, default=1.0,
                        help='softmax temperature over centroid cosine sims (->0 = hard top-1)')
    parser.add_argument('--combine', type=str, default='prob', choices=['prob', 'logit'])
    parser.add_argument('--max-new-tokens', type=int, default=0,
                        help='0 = use the per-dataset default from ds_collections')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--out-dir', type=str, default='results_soft_moe')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    evaluate(args)
