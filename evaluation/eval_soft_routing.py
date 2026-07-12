"""Generic SOFT-ROUTED (per-token expert MIXTURE) evaluation across the VQA benchmarks.

Unlike hard routing (each question -> argmax expert; evaluation/eval_hard_routing.py), soft routing
answers every question with a per-token MIXTURE of ALL experts, weighted by a temperature-scaled
gate over the cluster centroids:

    w_k(x) = softmax_k( T * s_k(x) ),   s_k(x) = cos( normalize(f_x - mean), normalize(centroid_k) )

f_x = the image's CLIP ViT-B/16 feature in the mean-subtracted space -- identical routing math to
clustering/partition_jsonl_by_balanced_kmeans and eval_hard_routing. The weights are fixed per
example (they depend only on the image) and mix the experts' NEXT-TOKEN distributions at every
greedy decoding step (the generation-time analogue of the gating mixture p(x)=Σ_k w_k(x)·p_k(x)):

    p(token_t | context) = Σ_k w_k(x) · p_k(token_t | context),   token_t = argmax p(token_t)

All experts decode the SAME shared sequence (the mixture's greedy tokens), each advancing its own
KV cache. T is MULTIPLIED with the gate logits (T->0 uniform mixture, T->inf == hard routing).
Greedy, min_new_tokens=1, single 448 tile (never dynamic; see CLAUDE.md).

This is the generic successor to evaluation/eval_soft_routing_textvqa.py -- it reuses that script's
mixture decoder but parametrises the per-benchmark prompt / max_new_tokens / prediction format
(exactly matching eval/vqa/evaluate_vqa.py and, for refcoco, eval/refcoco/evaluate_grounding.py) and
reuses eval_hard_routing's own scorers, so the metric can never drift. Supported: the 7 vqa-runner
benchmarks (textvqa, vqav2, chartqa, ai2d, docvqa, infovqa, gqa), refcoco (8 RefCOCO/+/g splits,
grounding prompt, P@1 IoU@0.5, headline = mean of the 8), scienceqa (image-only subset, hint +
question + lettered choices, letter exact-match), pope (yes/no, Overall F1 via eval_pope.py,
image_root-joined basenames), and mme (bespoke non-jsonl path run_mme_soft: every question is
mixture-decoded, per-category prediction files scored by eval/mme/calculation.py -> perception +
cognition). All 11 benchmarks that hard routing covers are now soft-decodable.

Example:
    python evaluation/eval_soft_routing.py --benchmark chartqa --temperature 8.0 \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --expert-checkpoints <ckpt-cluster-0> <ckpt-cluster-1>

Run from the repo root with the venv active. --expert-checkpoints must be in CLUSTER ORDER
(expert k <-> centroid k). --limit N evaluates only the first N questions per split (smoke test).

Batched: --batch-size B decodes B questions at once (left-padded like model.batch_chat), mirroring
the dense evaluator's batching (see CLAUDE.md). The mixture decoder is hand-rolled (HF .generate
can't mix K models), so the batching is hand-rolled too: one batched prefill per expert amortises
the (dominant) image/prompt prefix across the batch, and every greedy step advances all B rows at
once. Each expert keeps its own (B, ...) KV cache, so peak memory ~ K * (dense batch of size B);
the batch-size ceiling is roughly the dense one divided by #experts. Default B=1 == legacy
per-sample decoding. Still K forwards/token; heavy for vqav2 (214k q), use --limit there.

Batched greedy is NOT bit-identical to B=1 (left-pad + batched-matmul reduction order flips
near-tie argmaxes) -- keep --batch-size fixed across any head-to-head comparison (see CLAUDE.md).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# reuse the exact routing (CLIP + centroids + mean subtraction), the benchmark registry, and the
# repo's own scorers -- so soft routing scores identically to hard routing / the upstream evaluator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hard_routing import (BENCHMARKS, CHAT, MME_IMAGES, MME_QUESTIONS,  # noqa: E402
                               REPO, clip_features, load_centroids, score)

if str(CHAT) not in sys.path:                      # the `internvl` package lives under internvl_chat/
    sys.path.insert(0, str(CHAT))

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PROMPT = 'Answer the question using a single word or phrase.'
# RefCOCO grounding prompt, verbatim from eval/refcoco/evaluate_grounding.py ('{}' <- the referring
# sentence). Unlike the vqa prompts (a suffix appended to the question), this is a template that
# WRAPS the sentence, so refcoco sets grounding=True (see row_user_text).
REFCOCO_PROMPT = 'Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>'
# ScienceQA non-CoT instruction, verbatim from eval/scienceqa/evaluate_scienceqa.py (args.cot off).
SQA_PROMPT = "Answer with the option's letter from the given choices directly."
SQA_LETTERS = ['A', 'B', 'C', 'D', 'E']
IMG_START, IMG_END, IMG_CONTEXT = '<img>', '</img>', '<IMG_CONTEXT>'

# Per-benchmark generation config, matching eval/vqa/evaluate_vqa.py (and, for refcoco/scienceqa/
# pope, the corresponding eval/*/evaluate_*.py) exactly:
#   prompt          -- suffix appended to the question ('' for ai2d, base for the rest); refcoco
#                      (grounding) WRAPS the sentence; scienceqa uses it as the letter instruction;
#                      pope's text field already carries the base prompt (prompt='')
#   max_new_tokens  -- 10 short-answer, 100 for chart/doc/info, refcoco boxes, scienceqa, pope
#   fmt             -- prediction dict shape each scorer expects (see build_pred)
#   question_key    -- row field holding the text ('question'; refcoco 'sent'; pope 'text')
#   grounding       -- refcoco only: wrap the sentence with `prompt` instead of appending a suffix
SOFT = {
    'textvqa':   dict(prompt=BASE_PROMPT,    max_new_tokens=10,  fmt='vqa'),
    'vqav2':     dict(prompt=BASE_PROMPT,    max_new_tokens=10,  fmt='vqa'),
    'chartqa':   dict(prompt=BASE_PROMPT,    max_new_tokens=100, fmt='chartqa'),
    'ai2d':      dict(prompt='',             max_new_tokens=10,  fmt='ai2d'),
    'docvqa':    dict(prompt=BASE_PROMPT,    max_new_tokens=100, fmt='anls'),
    'infovqa':   dict(prompt=BASE_PROMPT,    max_new_tokens=100, fmt='anls'),
    'gqa':       dict(prompt=BASE_PROMPT,    max_new_tokens=10,  fmt='anls'),
    'refcoco':   dict(prompt=REFCOCO_PROMPT, max_new_tokens=100, fmt='refcoco',
                      question_key='sent', grounding=True),
    'scienceqa': dict(prompt=SQA_PROMPT,     max_new_tokens=100, fmt='scienceqa'),
    'pope':      dict(prompt='',             max_new_tokens=100, fmt='pope', question_key='text'),
    # MME uses a bespoke non-jsonl path (run_mme_soft), not evaluate_split; prompt/max_new_tokens
    # here are informational (run_mme_soft hardcodes the base prompt + 20 to match eval/mme/eval.py).
    'mme':       dict(prompt=BASE_PROMPT,    max_new_tokens=20,  fmt='mme'),
}


def post_process(pred, option):
    """Extract the chosen option letter, verbatim from eval/scienceqa/evaluate_scienceqa.post_process
    (copied to avoid importing the whole internvl model stack)."""
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) > 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) > 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k
    if len(pred) > 1 and pred[1] == '.':
        pred = pred[0]
    if len(pred) > 1 and pred[0] == '(' and pred[2] == ')':
        pred = pred[1]
    return pred


def build_sqa_question(row, instruction):
    """ScienceQA user text: (hint +) question + lettered choices + the letter instruction --
    replicates eval/scienceqa/evaluate_scienceqa.ScienceQADataset.__getitem__ exactly."""
    q = row['question']
    if row.get('hint'):
        q = row['hint'] + '\n' + q
    choice_txt = '\n'.join(f'{SQA_LETTERS[i]}. {c}' for i, c in enumerate(row['choices']))
    return (q + '\n' + choice_txt + '\n' + instruction).strip()


def row_user_text(cfg, row):
    """The user-turn text for one row. refcoco (grounding) wraps the sentence; scienceqa builds the
    choice prompt; pope's `text` field already carries the base prompt; otherwise question+suffix."""
    if cfg['fmt'] == 'scienceqa':
        return build_sqa_question(row, cfg['prompt'])
    text = row[cfg.get('question_key', 'question')]
    if cfg['fmt'] == 'pope':                             # text already includes the base prompt
        return text.strip()
    if cfg.get('grounding'):
        return cfg['prompt'].format(text)
    return text + (' ' + cfg['prompt'] if cfg['prompt'] else '')


def build_pred(fmt, row, answer):
    """Prediction dict in exactly the shape the upstream evaluator writes for this benchmark, so
    eval_hard_routing.score() consumes it unchanged."""
    if fmt == 'refcoco':                                 # P@1: box text + gt box + (h, w) for denorm
        return {'answer': answer, 'gt_bbox': row['bbox'], 'hw': (row['height'], row['width'])}
    if fmt == 'scienceqa':                               # sqa: map generation -> option letter
        options = {SQA_LETTERS[i]: c for i, c in enumerate(row['choices'])}
        return {'answer': post_process(answer, options), 'gt_answers': SQA_LETTERS[row['answer']]}
    if fmt == 'pope':                                    # F1 scorer keys on question_id + 'text'
        return {'question_id': row['question_id'], 'text': answer}
    q, qid, ann = row['question'], row['question_id'], row.get('answer')
    if fmt == 'vqa':                                     # textvqa, vqav2
        return {'question': q, 'question_id': qid, 'answer': answer}
    if fmt == 'chartqa':                                 # relaxed_accuracy needs answer + annotation
        return {'question': q, 'answer': answer, 'annotation': ann}
    if fmt == 'ai2d':                                    # exact_match; id key is 'image'
        return {'question': q, 'image': qid, 'answer': answer, 'annotation': ann}
    if fmt == 'anls':                                    # docvqa, infovqa, gqa
        return {'question': q, 'questionId': qid, 'answer': answer, 'annotation': ann}
    raise SystemExit(f'unknown prediction fmt {fmt}')


# --------------------------------------------------------------------------- gating weights
def _softmax(x, axis=-1):
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
    weights = _softmax(temperature * (feats @ cents.T), axis=1)     # (n_images, K)
    return dict(zip(images, weights))


# --------------------------------------------------------------------------- model loading
def load_experts(checkpoints):
    from internvl.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer
    models, tok = [], None
    for c in checkpoints:
        p = Path(c)
        if not p.is_absolute():
            p = REPO / p
        if not (p / 'config.json').is_file():
            raise SystemExit(f'no config.json in checkpoint {p}')
        models.append(InternVLChatModel.from_pretrained(
            str(p), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval().cuda())
        if tok is None:                                # experts share the base tokenizer
            tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True, use_fast=False)
    return models, tok


def build_transform(image_size):
    from internvl.train.dataset import build_transform as _bt
    return _bt(is_train=False, input_size=image_size)


def build_query(model, user_text):
    """Replicate InternVLChatModel.chat prompt construction (single 448 tile, non-dynamic) for an
    already-assembled user-turn `user_text` (see row_user_text: question+suffix, or the refcoco
    grounding template). Matches evaluate_vqa.py / evaluate_grounding.py prompt construction."""
    from internvl.conversation import get_conv_template
    tmpl = get_conv_template(model.template)
    tmpl.system_message = model.system_message
    tmpl.append_message(tmpl.roles[0], '<image>\n' + user_text)
    tmpl.append_message(tmpl.roles[1], None)
    query = tmpl.get_prompt()
    image_tokens = IMG_START + IMG_CONTEXT * model.num_image_token + IMG_END  # num_patches = 1
    return query.replace('<image>', image_tokens, 1), tmpl.sep.strip()


# --------------------------------------------------------------------------- mixture decoding
def _backbone_step(m, inputs_embeds, attention_mask, position_ids, past_key_values):
    """Run the LM backbone and apply lm_head to ONLY the last position -> (past_key_values,
    last-token logits (B, vocab)). Calling the full language_model instead would materialize
    (B, L, vocab) float32 logits over the whole prompt, which OOMs on long prompts (e.g. scienceqa)
    at large batch. The last-token logits are identical either way."""
    out = m.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 use_cache=True)
    logits = m.language_model.lm_head(out.last_hidden_state[:, -1, :]).float()      # (B, vocab)
    return out.past_key_values, logits


@torch.no_grad()
def soft_generate(models, weights, tokenizer, pixel_values, input_ids, attention_mask,
                  eos_id, img_context_id, max_new_tokens):
    """Batched greedy decode mixing per-token probabilities of all experts by fixed per-row
    weights `weights` (B, K). `input_ids`/`attention_mask` are LEFT-padded (B, L) exactly as
    model.batch_chat builds them; `pixel_values` is (B, 3, H, W) (one 448 tile per row). Returns
    a list of B decoded answer strings. B=1 reduces to the legacy per-sample path."""
    device = pixel_values.device
    B, L = input_ids.shape
    K = len(models)
    # left-padded position ids (padding masked out), matching HF prepare_inputs_for_generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # ----- prefix pass: build each expert's image-injected embeds, prime its (B, ...) KV cache
    pkv, last_logits = [None] * K, [None] * K
    for k, m in enumerate(models):
        emb = m.language_model.get_input_embeddings()(input_ids).clone()      # (B, L, C)
        C = emb.shape[-1]
        vit = m.extract_feature(pixel_values).reshape(-1, C).to(emb.dtype)    # (B*num_img_tok, C)
        flat = emb.reshape(-1, C)
        sel = (input_ids.reshape(-1) == img_context_id)
        if int(sel.sum()) != vit.shape[0]:
            raise SystemExit(f'expert {k}: {int(sel.sum())} image slots but {vit.shape[0]} '
                             'vision tokens -- prompt/feature mismatch')
        flat[sel] = vit
        pkv[k], last_logits[k] = _backbone_step(m, flat.reshape(B, L, C), attention_mask,
                                                position_ids, None)                  # (B, vocab)

    # ----- mixed greedy loop, all B rows in lockstep
    generated = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    attn = attention_mask
    next_pos = position_ids[:, -1:] + 1                # (B, 1); position of the first new token
    for step in range(max_new_tokens):
        probs = sum(weights[:, k:k + 1] * torch.softmax(last_logits[k], dim=-1) for k in range(K))
        if step == 0:                                  # min_new_tokens = 1: never emit EOS first
            probs[:, eos_id] = -1.0
        nxt = probs.argmax(-1)                          # (B,)
        is_eos = nxt == eos_id
        for b in torch.nonzero((~finished) & (~is_eos)).flatten().tolist():
            generated[b].append(int(nxt[b]))
        finished = finished | is_eos
        if bool(finished.all()) or step == max_new_tokens - 1:
            break
        attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
        tok = nxt.unsqueeze(1)                          # (B, 1); finished rows fed a dummy, ignored
        for k, m in enumerate(models):
            emb = m.language_model.get_input_embeddings()(tok)
            pkv[k], last_logits[k] = _backbone_step(m, emb, attn, next_pos, pkv[k])
        next_pos = next_pos + 1

    texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    return [t.split('<|im_end|>')[0].strip() for t in texts]


# --------------------------------------------------------------------------- batched image loading
def resolve_image(image_root, image_key, row):
    """Path (relative to internvl_chat/) of a row's image, joining `image_root` when the jsonl
    stores a bare basename (POPE). Used identically for CLIP routing and image loading."""
    p = row[image_key]
    return str(Path(image_root) / p) if image_root else p


class SoftDataset(Dataset):
    """Parallel image load + transform per row; carries the assembled user text, gate weights, and
    raw row (the row is needed by build_pred: question_id for vqa, bbox/hw for refcoco)."""
    def __init__(self, rows, image_key, transform, img2w, cfg, image_root):
        self.rows, self.image_key, self.transform = rows, image_key, transform
        self.img2w, self.cfg, self.image_root = img2w, cfg, image_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rp = resolve_image(self.image_root, self.image_key, r)
        img = Image.open(CHAT / rp).convert('RGB')
        return {'pixel_values': self.transform(img),                 # (3, H, W)
                'text': row_user_text(self.cfg, r), 'row': r,
                'weights': self.img2w[rp]}                           # (K,)


def collate_fn(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])   # (B, 3, H, W)
    texts = [b['text'] for b in batch]
    rows = [b['row'] for b in batch]
    weights = np.stack([b['weights'] for b in batch])                # (B, K)
    return pixel_values, texts, rows, weights


def decode_texts(models, tokenizer, pixel_values, texts, weights, eos_id, img_context_id,
                 max_new_tokens, device):
    """Left-pad + tokenize the assembled per-row user texts and mixture-decode one batch. Shared by
    the jsonl split driver (evaluate_split) and the MME path so both decode identically."""
    tokenizer.padding_side = 'left'                    # left-pad so the greedy step aligns per row
    queries = [build_query(models[0], t)[0] for t in texts]
    enc = tokenizer(queries, return_tensors='pt', padding=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    w = torch.tensor(np.asarray(weights), dtype=torch.float32, device=device)       # (B, K)
    return soft_generate(models, w, tokenizer, pixel_values.to(torch.bfloat16).to(device),
                         input_ids, attention_mask, eos_id, img_context_id, max_new_tokens)


# --------------------------------------------------------------------------- per-split driver
def evaluate_split(ds_name, test_rel, bm, cfg, models, tokenizer, transform, eos_id,
                   img_context_id, centroids, mean_vector, args):
    rows = [json.loads(l) for l in open(CHAT / test_rel)]
    if args.limit:
        rows = rows[:args.limit]
    image_root = bm.image_root
    images = sorted({resolve_image(image_root, args.image_key, r) for r in rows})
    prompt_desc = 'grounding' if cfg.get('grounding') else ('<empty>' if not cfg['prompt'] else 'base')
    print(f'\n{ds_name}: {len(rows)} questions over {len(images)} images '
          f'(prompt={prompt_desc}, '
          f'max_new_tokens={cfg["max_new_tokens"]}, batch_size={args.batch_size})')

    img2w = gate_weights(images, centroids, mean_vector, args.temperature, args)
    avg_w = np.mean(np.stack([img2w[i] for i in images]), axis=0)
    print('  mean gate: ' + ', '.join(f'e{k}={avg_w[k]:.3f}' for k in range(len(models))))

    device = next(models[0].parameters()).device
    loader = DataLoader(SoftDataset(rows, args.image_key, transform, img2w, cfg, image_root),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    preds, done = [], 0
    for pixel_values, texts, batch_rows, weights in loader:
        answers = decode_texts(models, tokenizer, pixel_values, texts, weights, eos_id,
                               img_context_id, cfg['max_new_tokens'], device)
        for r, ans in zip(batch_rows, answers):
            preds.append(build_pred(cfg['fmt'], r, ans))
        done += len(batch_rows)
        if done % max(args.batch_size, 200) < args.batch_size or done == len(rows):
            print(f'  {done}/{len(rows)}', flush=True)

    pred_file = CHAT / args.out_dir / f'{ds_name}_softroute_T{args.temperature}.json'
    json.dump(preds, open(pred_file, 'w'))
    acc = score(preds, bm, args, is_union=True)
    print(f'  {ds_name}: {100 * acc:.2f}%   (preds -> {pred_file.relative_to(CHAT)})')
    return {'ds_name': ds_name, 'n_questions': len(preds),
            'mean_gate_weights': avg_w.tolist(), 'accuracy': acc}


# --------------------------------------------------------------------------- MME (bespoke path)
def read_mme_questions():
    """{category: [(img, question, gt)]} from eval/mme/Your_Results/*.txt. Two questions per image
    on adjacent lines; calculation.py chunks each category file into consecutive pairs, so ORDER
    within a category must be preserved when writing predictions."""
    cats = {}
    for f in sorted((CHAT / MME_QUESTIONS).glob('*.txt')):
        rows = []
        for line in open(f):
            if not line.strip():
                continue
            img, question, gt = line.rstrip('\n').split('\t')
            rows.append((img, question, gt))
        cats[f.stem] = rows
    return cats


def mme_post_processing(response):
    """Verbatim from eval/mme/eval.post_processing so calculation.py's yes/no parse is unchanged."""
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    return re.sub(re.compile(r'[一-龥]'), '', response)


@torch.no_grad()
def run_mme_soft(models, tokenizer, transform, eos_id, img_context_id, centroids, mean_vector, args):
    """Soft-mixture MME. Unlike hard routing (which splits questions across experts), EVERY question
    is mixture-decoded. Replicates eval/mme/eval.py's prompt (question + base prompt), single 448
    tile, max_new_tokens=20, and 4-field output line (img<TAB>question<TAB>gt<TAB>pred), preserving
    per-category order; then scores Perception+Cognition with eval/mme/calculation.py on the union."""
    work = CHAT / args.out_dir
    preds_dir = work / 'predictions'
    preds_dir.mkdir(parents=True, exist_ok=True)
    cats = read_mme_questions()
    total = sum(len(v) for v in cats.values())

    keys = sorted({(cat, img) for cat, rows in cats.items() for img, _, _ in rows})
    rel_paths = [str(Path(MME_IMAGES) / cat / img) for cat, img in keys]
    print(f'\nMME: {total} questions over {len(keys)} images across {len(cats)} categories '
          f'(prompt=base, max_new_tokens=20, batch_size={args.batch_size})')
    path2w = gate_weights(rel_paths, centroids, mean_vector, args.temperature, args)
    avg_w = np.mean(np.stack([path2w[p] for p in rel_paths]), axis=0)
    print('  mean gate: ' + ', '.join(f'e{k}={avg_w[k]:.3f}' for k in range(len(models))))

    device = next(models[0].parameters()).device
    done = 0
    for cat, rows in cats.items():
        lines_out = []
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            rels = [str(Path(MME_IMAGES) / cat / img) for img, _, _ in batch]
            pv = torch.stack([transform(Image.open(CHAT / r).convert('RGB')) for r in rels])
            texts = [q + ' ' + BASE_PROMPT for _, q, _ in batch]
            weights = np.stack([path2w[r] for r in rels])
            answers = decode_texts(models, tokenizer, pv, texts, weights, eos_id,
                                   img_context_id, 20, device)
            for (img, q, gt), ans in zip(batch, answers):
                lines_out.append(f'{img}\t{q} {BASE_PROMPT}\t{gt}\t{mme_post_processing(ans)}\n')
            done += len(batch)
        open(preds_dir / f'{cat}.txt', 'w').writelines(lines_out)
        print(f'  {cat}: {len(rows)}  ({done}/{total})', flush=True)

    env = dict(os.environ)
    env['PYTHONPATH'] = f"{CHAT}:{env.get('PYTHONPATH', '')}"
    out = subprocess.run([sys.executable, 'calculation.py', '--results_dir', str(preds_dir)],
                         cwd=CHAT / 'eval/mme', env=env, capture_output=True, text=True)
    print(out.stdout)
    section, totals = None, {}
    for line in out.stdout.splitlines():
        if 'Perception' in line:
            section = 'perception'
        elif 'Cognition' in line:
            section = 'cognition'
        elif 'total score' in line and section:
            totals[section] = float(line.split(':')[1].strip())
    if 'perception' not in totals or 'cognition' not in totals:
        raise SystemExit(f'could not parse MME scores:\n{out.stdout}\n{out.stderr}')
    combined = totals['perception'] + totals['cognition']

    print('\n' + '=' * 66)
    print(f'SOFT-ROUTED MME  (T={args.temperature}, perception+cognition, {len(models)} experts)')
    print('=' * 66)
    print(f'  Perception: {totals["perception"]:.2f}   Cognition: {totals["cognition"]:.2f}')
    print(f'  MME TOTAL : {combined:.2f}')
    summary = {'benchmark': 'mme', 'routing': 'soft', 'temperature': args.temperature,
               'metric': 'perception+cognition', 'clustering_dir': args.clustering_dir,
               'expert_checkpoints': list(args.expert_checkpoints),
               'mean_gate_weights': avg_w.tolist(), 'n_questions': total,
               'perception': totals['perception'], 'cognition': totals['cognition'],
               'mme_total': combined}
    dest = work / f'summary_softroute_mme_T{args.temperature}.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', required=True, choices=sorted(SOFT))
    ap.add_argument('--temperature', type=float, required=True,
                    help='gate temperature T (softmax multiplier); T->inf recovers hard routing')
    ap.add_argument('--clustering-dir', required=True)
    ap.add_argument('--expert-checkpoints', nargs='+', required=True,
                    help='expert checkpoint dirs IN CLUSTER ORDER (expert k <-> centroid k)')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--out-dir', default=None, help='relative to internvl_chat/ '
                                                    '(default results/softroute/<benchmark>)')
    ap.add_argument('--image-key', default='image')
    ap.add_argument('--clip-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--clip-batch-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=1,
                    help='questions decoded at once (left-padded mixture). 1 = legacy per-sample; '
                         'each expert holds a (B, ...) KV cache so peak mem ~ K*B -- keep <= dense '
                         'ceiling / #experts. Match this to the baseline you compare against.')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='first N questions per split (smoke test)')
    ap.add_argument('--splits', nargs='+', default=None,
                    help='restrict to these ds_name splits (e.g. refcoco_val for a single RefCOCO '
                         'split); default = all of the benchmark\'s splits')
    args = ap.parse_args()

    cfg = SOFT[args.benchmark]
    bm = BENCHMARKS.get(args.benchmark)                # None for mme (bespoke, not in the registry)
    if args.out_dir is None:
        args.out_dir = f'results/softroute/{args.benchmark}'
    (CHAT / args.out_dir).mkdir(parents=True, exist_ok=True)

    # preflight (before loading K models): MME needs its bespoke files; every other benchmark
    # needs its test jsonls + annotation.
    is_mme = args.benchmark == 'mme'
    if is_mme:
        for need in [MME_QUESTIONS, MME_IMAGES, 'eval/mme/calculation.py']:
            if not (CHAT / need).exists():
                raise SystemExit(f'MME data missing: internvl_chat/{need}')
    else:
        splits = [(bm.ds_name, bm.test)] + bm.extra_tests
        if args.splits:                                # restrict to requested ds_name splits
            keep = set(args.splits)
            chosen = [(ds, t) for ds, t in splits if ds in keep]
            if not chosen:
                raise SystemExit(f'--splits {args.splits} matched none of '
                                 f'{[ds for ds, _ in splits]}')
            splits = chosen
        needed = [t for _, t in splits] + ([bm.annotation] if bm.annotation else []) + bm.needs
        missing = [f for f in needed if f and not (CHAT / f).exists()]
        if missing:
            raise SystemExit('missing eval data:\n'
                             + '\n'.join(f'  - internvl_chat/{m}' for m in missing))

    n_experts = len(args.expert_checkpoints)
    metric = bm.metric if bm else 'perception+cognition'
    print(f'benchmark={args.benchmark}  routing=soft  T={args.temperature}  metric={metric}  '
          f'experts={n_experts}')
    centroids, mean_vector = load_centroids(args, n_experts)

    models, tokenizer = load_experts(args.expert_checkpoints)
    img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    transform = build_transform(models[0].config.force_image_size
                                or models[0].config.vision_config.image_size)
    _, sep = build_query(models[0], 'x')
    eos_id = tokenizer.convert_tokens_to_ids(sep)

    if is_mme:
        run_mme_soft(models, tokenizer, transform, eos_id, img_context_id,
                     centroids, mean_vector, args)
        return

    results = [evaluate_split(ds, test, bm, cfg, models, tokenizer, transform, eos_id,
                              img_context_id, centroids, mean_vector, args)
               for ds, test in splits]

    print('\n' + '=' * 66)
    print(f'SOFT-ROUTED {args.benchmark.upper()}  (T={args.temperature}, {bm.metric}, '
          f'{n_experts} experts)')
    print('=' * 66)
    for r in results:
        print(f'  {r["ds_name"]:<26} n={r["n_questions"]:6d}   {100 * r["accuracy"]:6.2f}%')
    headline = sum(r['accuracy'] for r in results) / len(results)
    if len(results) > 1:
        print(f'  {"headline (mean of splits)":<26}          {100 * headline:6.2f}%')

    summary = {'benchmark': args.benchmark, 'routing': 'soft', 'temperature': args.temperature,
               'metric': bm.metric, 'clustering_dir': args.clustering_dir,
               'expert_checkpoints': list(args.expert_checkpoints),
               'splits': results, 'headline_accuracy': headline}
    dest = CHAT / args.out_dir / f'summary_softroute_{args.benchmark}_T{args.temperature}.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')


if __name__ == '__main__':
    main()
