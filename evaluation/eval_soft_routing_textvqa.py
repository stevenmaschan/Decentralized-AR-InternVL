"""End-to-end SOFT-ROUTED evaluation of cluster experts on TextVQA.

Unlike hard routing (each question -> argmax expert), soft routing answers every question with a
per-token MIXTURE of all experts, weighted by a temperature-scaled gate over the cluster
centroids:

    w_k(x) = softmax_k( T * s_k(x) ),   s_k(x) = cos( normalize(f_x - mean), normalize(centroid_k) )

where f_x is the image's CLIP ViT-B/16 feature (mean-subtracted space -- identical routing math to
clustering/partition_jsonl_by_balanced_kmeans and evaluation/eval_hard_routing). The weights are
fixed per example (they depend only on the image) and are used to mix the experts' NEXT-TOKEN
distributions at every decoding step -- the generation-time analogue of the gating mixture
p(x) = Σ_k w_k(x)·p_k(x) (see CLAUDE.md "Gating"):

    p(token_t | context) = Σ_k w_k(x) · p_k(token_t | context),   token_t = argmax_token p(token_t)

All experts decode the SAME shared sequence (the mixture's greedy tokens), each advancing its own
KV cache through its own weights. T is MULTIPLIED with the gate logits (T->0 uniform mixture,
T->inf hard argmax == hard routing). Greedy, min_new_tokens=1, max_new_tokens=10 (matches
eval/vqa/evaluate_vqa.py for textvqa_val). Never uses dynamic tiling (single 448 tile; see
CLAUDE.md).

Scoring reuses the repo's own TextVQA accuracy metric (textvqa_eval.TextVQAAccuracyEvaluator).

Example:
    python evaluation/eval_soft_routing_textvqa.py --temperature 8.0 \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --expert-checkpoints \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-0 \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-1

Run from the repo root with the venv active. --expert-checkpoints must be in CLUSTER ORDER
(expert k <-> centroid k). --limit N evaluates only the first N questions (smoke test).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

# reuse the exact routing implementation (CLIP features + centroid loading + mean subtraction)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hard_routing import CHAT, REPO, clip_features, load_centroids  # noqa: E402

# the `internvl` package lives under internvl_chat/ (checkpoints have no trust_remote_code files)
if str(CHAT) not in sys.path:
    sys.path.insert(0, str(CHAT))

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_JSONL = 'data/textvqa/textvqa_val.jsonl'
ANNOTATION = 'data/textvqa/textvqa_val_annotations.json'
PROMPT = 'Answer the question using a single word or phrase.'
IMG_START, IMG_END, IMG_CONTEXT = '<img>', '</img>', '<IMG_CONTEXT>'
MAX_NEW_TOKENS = 10


# --------------------------------------------------------------------------- gating weights
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gate_weights(images, centroids, mean_vector, temperature, args):
    """w_k(image) = softmax_k(T * cos(image, centroid_k)) in the mean-subtracted space."""
    feats = clip_features(images, args.clip_model, args.clip_batch_size, args.num_workers)
    if mean_vector is not None:
        feats = feats - mean_vector
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    cents = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sims = feats @ cents.T                       # cosine similarity, (n_images, K)
    weights = softmax(temperature * sims, axis=1)
    return dict(zip(images, weights)), sims


# --------------------------------------------------------------------------- model loading
def load_experts(checkpoints):
    from internvl.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer
    models = []
    tok = None
    for c in checkpoints:
        p = Path(c)
        if not p.is_absolute():
            p = REPO / p
        if not (p / 'config.json').is_file():
            raise SystemExit(f'no config.json in checkpoint {p}')
        m = InternVLChatModel.from_pretrained(
            str(p), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval().cuda()
        models.append(m)
        if tok is None:  # experts share the base tokenizer; one is enough
            tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True, use_fast=False)
    return models, tok


def build_transform(image_size):
    from internvl.train.dataset import build_transform as _bt
    return _bt(is_train=False, input_size=image_size)


def build_query(model, question):
    """Replicate InternVLChatModel.chat prompt construction (single 448 tile, non-dynamic)."""
    from internvl.conversation import get_conv_template
    tmpl = get_conv_template(model.template)
    tmpl.system_message = model.system_message
    tmpl.append_message(tmpl.roles[0], '<image>\n' + question + ' ' + PROMPT)
    tmpl.append_message(tmpl.roles[1], None)
    query = tmpl.get_prompt()
    image_tokens = IMG_START + IMG_CONTEXT * model.num_image_token + IMG_END  # num_patches = 1
    return query.replace('<image>', image_tokens, 1), tmpl.sep.strip()


# --------------------------------------------------------------------------- mixture decoding
@torch.no_grad()
def soft_generate(models, weights, tokenizer, pixel_values, query, eos_id, img_context_id):
    """Greedy decode mixing per-token probabilities of all experts by fixed weights `weights`."""
    device = pixel_values.device
    input_ids = tokenizer(query, return_tensors='pt')['input_ids'].to(device)
    L = input_ids.shape[1]
    K = len(models)

    # ----- prefix pass: build each expert's image-injected embeds, prime its KV cache
    pkv, last_logits = [None] * K, [None] * K
    for k, m in enumerate(models):
        emb = m.language_model.get_input_embeddings()(input_ids).clone()      # (1, L, C)
        C = emb.shape[-1]
        vit = m.extract_feature(pixel_values).reshape(-1, C).to(emb.dtype)    # (num_img_tok, C)
        flat = emb.reshape(-1, C)
        sel = (input_ids.reshape(-1) == img_context_id)
        if int(sel.sum()) != vit.shape[0]:
            raise SystemExit(f'expert {k}: {int(sel.sum())} image slots but {vit.shape[0]} '
                             'vision tokens -- prompt/feature mismatch')
        flat[sel] = vit
        out = m.language_model(inputs_embeds=flat.reshape(1, L, C),
                               attention_mask=torch.ones((1, L), device=device),
                               use_cache=True)
        pkv[k], last_logits[k] = out.past_key_values, out.logits[:, -1, :].float()

    # ----- mixed greedy loop
    generated = []
    for step in range(MAX_NEW_TOKENS):
        probs = sum(weights[k] * torch.softmax(last_logits[k], dim=-1) for k in range(K))
        if step == 0:                       # min_new_tokens = 1: never emit EOS first
            probs[0, eos_id] = -1.0
        nxt = int(probs.argmax(-1))
        if nxt == eos_id:
            break
        generated.append(nxt)
        cur_len = L + len(generated)        # position of the token we now feed = cur_len - 1
        tok = torch.tensor([[nxt]], device=device)
        pos = torch.tensor([[cur_len - 1]], device=device)
        attn = torch.ones((1, cur_len), device=device)
        for k, m in enumerate(models):
            emb = m.language_model.get_input_embeddings()(tok)
            out = m.language_model(inputs_embeds=emb, attention_mask=attn, position_ids=pos,
                                   past_key_values=pkv[k], use_cache=True)
            pkv[k], last_logits[k] = out.past_key_values, out.logits[:, -1, :].float()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.split('<|im_end|>')[0].strip()


# --------------------------------------------------------------------------- scoring
def score_textvqa(preds):
    for p in (str(CHAT), str(CHAT / 'eval/vqa')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from textvqa_eval import TextVQAAccuracyEvaluator  # noqa: E402
    ann = json.load(open(CHAT / ANNOTATION))['annotations']
    qid2gt = {a['question_id']: [x['answer'] for x in a['answers']] for a in ann}
    rows = [{'pred_answer': p['answer'], 'gt_answers': qid2gt[p['question_id']]} for p in preds]
    return TextVQAAccuracyEvaluator().eval_pred_list(rows)


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--temperature', type=float, required=True,
                    help='gate temperature T (softmax multiplier); T->inf recovers hard routing')
    ap.add_argument('--clustering-dir', required=True)
    ap.add_argument('--expert-checkpoints', nargs='+', required=True,
                    help='expert checkpoint dirs IN CLUSTER ORDER (expert k <-> centroid k)')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--out-dir', default='results/softroute/textvqa',
                    help='relative to internvl_chat/ (default results/softroute/textvqa)')
    ap.add_argument('--image-key', default='image')
    ap.add_argument('--clip-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--clip-batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='evaluate only the first N questions')
    args = ap.parse_args()

    # preflight
    for f in (TEST_JSONL, ANNOTATION):
        if not (CHAT / f).is_file():
            raise SystemExit(f'missing eval data: internvl_chat/{f}')
    n_experts = len(args.expert_checkpoints)
    centroids, mean_vector = load_centroids(args, n_experts)

    rows = [json.loads(l) for l in open(CHAT / TEST_JSONL)]
    if args.limit:
        rows = rows[:args.limit]
    images = sorted({r[args.image_key] for r in rows})
    for im in images[:1] + images[-1:]:
        if not (CHAT / im).is_file():
            raise SystemExit(f'missing image: internvl_chat/{im}')
    print(f'soft routing  T={args.temperature}  experts={n_experts}  '
          f'questions={len(rows)} over {len(images)} images')

    img2w, sims = gate_weights(images, centroids, mean_vector, args.temperature, args)
    # report the average gate (how soft the mixture is)
    avg_w = np.mean(np.stack([img2w[i] for i in images]), axis=0)
    print('  mean gate weights over images: '
          + ', '.join(f'expert{k}={avg_w[k]:.3f}' for k in range(n_experts)))

    models, tokenizer = load_experts(args.expert_checkpoints)
    img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    transform = build_transform(models[0].config.force_image_size
                                or models[0].config.vision_config.image_size)
    # eos from the template (same for both experts)
    _, sep = build_query(models[0], 'x')
    eos_id = tokenizer.convert_tokens_to_ids(sep)

    (CHAT / args.out_dir).mkdir(parents=True, exist_ok=True)
    preds = []
    for i, r in enumerate(rows, 1):
        img = Image.open(CHAT / r[args.image_key]).convert('RGB')
        px = transform(img).unsqueeze(0).to(torch.bfloat16).cuda()
        query, _ = build_query(models[0], r['question'])   # identical across experts
        w = img2w[r[args.image_key]].tolist()
        ans = soft_generate(models, w, tokenizer, px, query, eos_id, img_context_id)
        preds.append({'question': r['question'], 'question_id': r['question_id'], 'answer': ans})
        if i % 200 == 0 or i == len(rows):
            print(f'  {i}/{len(rows)}', flush=True)

    pred_file = CHAT / args.out_dir / f'textvqa_softroute_T{args.temperature}.json'
    json.dump(preds, open(pred_file, 'w'))
    accuracy = score_textvqa(preds)

    print('\n' + '=' * 60)
    print(f'SOFT-ROUTED TextVQA  (T={args.temperature}, {n_experts} experts)')
    print(f'  questions:   {len(preds)}')
    print(f'  mean gate:   ' + ', '.join(f'e{k}={avg_w[k]:.3f}' for k in range(n_experts)))
    print(f'  VQA accuracy: {100 * accuracy:.2f}%')
    print('=' * 60)

    summary = {'benchmark': 'textvqa', 'routing': 'soft', 'temperature': args.temperature,
               'clustering_dir': args.clustering_dir,
               'expert_checkpoints': list(args.expert_checkpoints),
               'n_questions': len(preds), 'mean_gate_weights': avg_w.tolist(),
               'accuracy': accuracy}
    dest = CHAT / args.out_dir / f'summary_softroute_textvqa_T{args.temperature}.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'saved -> {dest}\npreds -> {pred_file}')


if __name__ == '__main__':
    main()
