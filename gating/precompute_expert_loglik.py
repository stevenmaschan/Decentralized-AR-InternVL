#!/usr/bin/env python3
"""
Precompute per-sequence log-likelihoods under each cluster *expert* model.

This is step 1 of fitting the gating temperature (see fit_gating_temperature.py).
For every training sequence x (one JSONL line / conversation) and every expert
model k we compute the sum of token log-probs over the RESPONSE tokens
(labels != IGNORE_INDEX), i.e.

    logp_k(x) = sum_{t in response} log p_k(token_t | context_{<t}).

The mixture-of-experts marginal likelihood the temperature is fit against is

    p(x) = sum_k w_k(x) * p_k(x),    w_k(x) = softmax_k( T * s_k(x) ),

where s_k(x) is the cosine similarity of x's image to centroid k in the
mean-subtracted, L2-normalized space (matching single_stage_balanced_kmeans.py),
and T (temperature, MULTIPLIED) is the scalar we later fit.

This is FORWARD-ONLY (no backward/optimizer), one pass over UNIQUE sequences, so
it is a fraction of training cost. To actually realize that it (a) loads images
in parallel with a DataLoader, (b) runs a batched forward with a per-batch token
budget, and (c) checkpoints so a long run is resumable.

The expert checkpoints need NOT correspond to the clustering used for the gating
logits (they can be stand-ins for debugging). Reuses the exact training-time
tokenization / labeling / image transform via internvl_chat's LazySupervisedDataset,
calling its per-item preprocessing directly (we bypass __getitem__ because it
silently retries a RANDOM index on load error, breaking index<->image alignment).

Output: an .npz (row-aligned across all sequences of the meta, in file order) with
    loglik         (N, K_exp)  float64  sequence log-likelihood per expert (NaN if skipped)
    n_resp_tokens  (N,)        int32    # response tokens (same for all experts; -1 if skipped)
    repeat_time    (N,)        float32  dataset repeat factor (sequence weight)
    gating_logits  (N, K_clu)  float32  cosine sim to each centroid (NaN if no feature)
    has_gating     (N,)        bool
    done           (N,)        bool     sequence successfully evaluated
plus a <output>.json side-car (expert paths, dataset names, image paths/keys).

Usage (full dense set, real experts, resumable):
    /lambda/nfs/virginia/clip-feat-venv/bin/python gating/precompute_expert_loglik.py \
        --meta /lambda/nfs/virginia/internvl-data/dense/dataset_mixture.json \
        --expert-checkpoints /lambda/nfs/virginia/old-models/expert-0 /lambda/nfs/virginia/old-models/expert-1 \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --features-prefix clustering/unique_features_vit_base_patch16 \
        --num-workers 8 --max-batch-tokens 8192 --output gating/expert_loglik_dense.npz --resume
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'internvl_chat'))
sys.path.insert(0, str(REPO_ROOT / 'clustering'))

from internvl.train.internvl_chat_pretrain import LazySupervisedDataset       # noqa: E402
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,          # noqa: E402
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.model.internvl_chat import (InternVLChatConfig,                  # noqa: E402
                                          InternVLChatModel)
from internvl.patch import concat_pad_data_collator                            # noqa: E402
from transformers import AutoTokenizer                                         # noqa: E402

IGNORE_INDEX = -100
CE_ROW_CHUNK = 4096   # rows of (token, vocab) per cross-entropy chunk (bounds memory)


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def init_single_process_group():
    """LazySupervisedDataset.__init__ needs torch.distributed initialized."""
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '34599')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('LOCAL_RANK', '0')
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)


def build_tokenizer(tokenizer_path, max_seq_length, use_fast=False):
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=use_fast)
    tok.tokenizer_path = tokenizer_path
    tok.model_max_length = max_seq_length
    tok.add_tokens([IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                    QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                    REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN], special_tokens=True)
    return tok, tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


def load_expert(ckpt_path, args, img_context_token_id, device):
    config = InternVLChatConfig.from_pretrained(ckpt_path)
    config.vision_config.drop_path_rate = 0.0
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'
    config.template = args.conv_style
    config.select_layer = args.vision_select_layer
    config.dynamic_image_size = args.dynamic_image_size
    config.use_thumbnail = args.use_thumbnail
    config.ps_version = args.ps_version
    config.min_dynamic_patch = args.min_dynamic_patch
    config.max_dynamic_patch = args.max_dynamic_patch
    model = InternVLChatModel.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16, config=config)
    model.img_context_token_id = img_context_token_id
    model.eval().to(device)
    return model


def make_stub_dataset(root, max_dyn, tokenizer, num_image_token, args, scratch_dir):
    """A LazySupervisedDataset built on a 1-line stub (so construction is instant);
    only its preprocessing (transform / template / root / num_image_token) is used."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    stub = scratch_dir / f'stub_{abs(hash((root, max_dyn))) % (10**8)}.jsonl'
    stub.write_text('{"image": "", "conversations": [{"from": "human", "value": "x"}, '
                    '{"from": "gpt", "value": "y"}]}\n')
    meta_ds = {'root': root, 'annotation': str(stub), 'data_augment': False,
               'max_dynamic_patch': max_dyn, 'repeat_time': 1}
    return LazySupervisedDataset(
        args.conv_style, meta_ds, tokenizer, tcs_loader=None, ds_name='stub',
        num_image_token=num_image_token, image_size=args.force_image_size,
        is_train=False, pad2square=False, group_by_length=False,
        dynamic_image_size=args.dynamic_image_size, use_thumbnail=args.use_thumbnail,
        min_dynamic_patch=args.min_dynamic_patch, max_dynamic_patch=max_dyn,
        repeat_time=1, normalize_type='imagenet', use_packed_ds=False,
        data_rank=0, data_world_size=1, distributed_mode=False,
        force_shuffle=False, random_seed=0)


def get_item_for_line(ds, data_item):
    img = data_item.get('image')
    if img is not None and len(img) != 0:
        if isinstance(img, list):
            return ds.multi_modal_multi_image_get_item(data_item)
        return ds.multi_modal_get_item(data_item)
    if data_item.get('video'):
        return ds.video_get_item(data_item)
    return ds.pure_text_get_item(data_item)


def real_length(attention_mask):
    mask = attention_mask.bool()
    if not bool(mask.any()):
        return attention_mask.numel()
    return int(mask.nonzero().max().item()) + 1


def first_image_of(data_item):
    img = data_item.get('image')
    if not img:
        return None
    return img[0] if isinstance(img, list) else img


# --------------------------------------------------------------------------- #
# dataset over the manifest (runs in DataLoader workers -> parallel image load)
# --------------------------------------------------------------------------- #
class ManifestDataset(Dataset):
    """Preprocesses one manifest entry into model inputs (trimmed to real length)."""

    def __init__(self, manifest, order, ds_by_key):
        self.manifest = manifest        # list of dicts (gidx, ds_name, line, key)
        self.order = order              # positions (into manifest) to process this run
        self.ds_by_key = ds_by_key      # (root, max_dyn) -> LazySupervisedDataset

    def __len__(self):
        return len(self.order)

    def __getitem__(self, pos):
        gidx = self.order[pos]
        entry = self.manifest[gidx]
        try:
            data_item = json.loads(entry['line'])
            ds = self.ds_by_key[entry['key_cfg']]
            ret = get_item_for_line(ds, data_item)
            n = real_length(ret['attention_mask'])
            return {
                'input_ids': ret['input_ids'][:n],
                'labels': ret['labels'][:n],
                'attention_mask': ret['attention_mask'][:n],
                'pixel_values': ret['pixel_values'],
                'image_flags': ret['image_flags'],
                'gidx': gidx,
            }
        except Exception as e:
            return {'gidx': gidx, 'failed': f'{type(e).__name__}: {e}'}


def collate(features):
    valid = [f for f in features if 'failed' not in f]
    failed = [f['gidx'] for f in features if 'failed' in f]
    gidxs = [f['gidx'] for f in valid]
    batch = None
    if valid:
        for f in valid:
            f.pop('gidx', None)
        batch = concat_pad_data_collator(valid)
    return batch, gidxs, failed


class TokenBudgetSampler:
    """Yield batches of dataset positions with bounded total tokens (memory) —
    positions are pre-sorted by length so each batch has similar-length items."""

    def __init__(self, lengths_sorted_positions, est_tokens, max_tokens, max_items):
        self.positions = lengths_sorted_positions
        self.est = est_tokens
        self.max_tokens = max_tokens
        self.max_items = max_items

    def __iter__(self):
        batch, tok = [], 0
        for p in self.positions:
            t = self.est[p]
            if batch and (tok + t > self.max_tokens or len(batch) >= self.max_items):
                yield batch
                batch, tok = [], 0
            batch.append(p)
            tok += t
        if batch:
            yield batch

    def __len__(self):
        n, batch, tok = 0, 0, 0
        for p in self.positions:
            t = self.est[p]
            if batch and (tok + t > self.max_tokens or batch >= self.max_items):
                n += 1
                batch, tok = 0, 0
            batch += 1
            tok += t
        return n + (1 if batch else 0)


# --------------------------------------------------------------------------- #
# batched forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_loglik(model, batch, device):
    """Per-row summed log-prob over response tokens for a collated batch."""
    out = model(
        pixel_values=batch['pixel_values'].to(device=device, dtype=model.dtype),
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        image_flags=batch['image_flags'].reshape(-1, 1).to(device),
        labels=None, use_cache=False)
    V = out.logits.size(-1)
    shift_logits = out.logits[:, :-1, :].reshape(-1, V)
    shift_labels = batch['labels'][:, 1:].reshape(-1).to(device)
    # chunked cross-entropy keeps the fp32 logit copy small regardless of batch size
    nll = torch.empty(shift_labels.shape, device=device, dtype=torch.float32)
    for s in range(0, shift_labels.numel(), CE_ROW_CHUNK):
        e = s + CE_ROW_CHUNK
        nll[s:e] = F.cross_entropy(shift_logits[s:e].float(), shift_labels[s:e],
                                   ignore_index=IGNORE_INDEX, reduction='none')
    B, Lm1 = batch['labels'].size(0), batch['labels'].size(1) - 1
    return (-nll.view(B, Lm1).sum(dim=1)).cpu().numpy()   # (B,)


def n_resp_of(batch):
    sl = batch['labels'][:, 1:]
    return (sl != IGNORE_INDEX).sum(dim=1).cpu().numpy().astype(np.int32)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--expert-checkpoints', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--clustering-dir', default=None)
    ap.add_argument('--features-prefix', default='clustering/unique_features_vit_base_patch16')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--max-per-dataset', type=int, default=None)
    ap.add_argument('--sample-frac', type=float, default=None,
                    help='randomly keep this fraction of EACH dataset (e.g. 0.1 for 10%%)')
    ap.add_argument('--sample-seed', type=int, default=42,
                    help='seed for --sample-frac (reproducible per dataset)')
    ap.add_argument('--limit', type=int, default=None)
    # throughput / memory
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--max-batch-tokens', type=int, default=8192,
                    help='per-batch token budget (bounds GPU memory)')
    ap.add_argument('--max-batch-size', type=int, default=48)
    ap.add_argument('--checkpoint-every', type=int, default=5000,
                    help='save partial results every N newly-evaluated sequences')
    ap.add_argument('--resume', action='store_true',
                    help='resume from <output>.partial.npz if present')
    # model / preprocessing config
    ap.add_argument('--conv-style', default='internvl2_5')
    ap.add_argument('--force-image-size', type=int, default=448)
    ap.add_argument('--max-seq-length', type=int, default=8192)
    ap.add_argument('--min-dynamic-patch', type=int, default=1)
    ap.add_argument('--max-dynamic-patch', type=int, default=12)
    ap.add_argument('--dynamic-image-size', action='store_true', default=True)
    ap.add_argument('--use-thumbnail', action='store_true', default=False)
    ap.add_argument('--ps-version', default='v2')
    ap.add_argument('--vision-select-layer', type=int, default=-1)
    ap.add_argument('--use-fast-tokenizer', action='store_true', default=False)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--scratch-dir', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    init_single_process_group()
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else Path(args.output).parent / '_stub_cache'

    tokenizer, img_context_token_id = build_tokenizer(
        args.expert_checkpoints[0], args.max_seq_length, args.use_fast_tokenizer)

    print(f'Loading {len(args.expert_checkpoints)} expert(s)...')
    experts = [load_expert(p, args, img_context_token_id, device)
               for p in args.expert_checkpoints]
    num_image_token = experts[0].num_image_token
    K_exp = len(experts)
    print(f'num_image_token={num_image_token}, K_exp={K_exp}')

    # ---- build the manifest (all lines, in file order) ----
    meta = json.loads(Path(args.meta).read_text())
    manifest, ds_by_key = [], {}
    for ds_name, meta_ds in meta.items():
        if args.datasets and ds_name not in args.datasets:
            continue
        root = meta_ds['root']
        max_dyn = int(meta_ds.get('max_dynamic_patch', args.max_dynamic_patch))
        key_cfg = (root, max_dyn)
        if key_cfg not in ds_by_key:
            ds_by_key[key_cfg] = make_stub_dataset(
                root, max_dyn, tokenizer, num_image_token, args, scratch_dir)
        rep = float(meta_ds.get('repeat_time', 1))
        with open(meta_ds['annotation']) as fh:
            lines = [ln for ln in fh if ln.strip()]
        if args.sample_frac is not None:
            # reproducible per-dataset random subset (stable across runs: crc32,
            # not the hash-randomized built-in hash())
            import zlib
            seed = (args.sample_seed * 1000003 + zlib.crc32(ds_name.encode())) % (2 ** 32)
            rng = np.random.default_rng(seed)
            k = max(1, round(len(lines) * args.sample_frac))
            sel = np.sort(rng.choice(len(lines), size=k, replace=False))
            lines = [lines[i] for i in sel]
        if args.max_per_dataset is not None:
            lines = lines[:args.max_per_dataset]
        for ln in lines:
            if args.limit is not None and len(manifest) >= args.limit:
                break
            try:
                img = first_image_of(json.loads(ln))
            except json.JSONDecodeError:
                continue
            manifest.append({'gidx': len(manifest), 'ds_name': ds_name, 'line': ln,
                             'repeat_time': rep, 'image': img or '',
                             'key_cfg': key_cfg})
        if args.limit is not None and len(manifest) >= args.limit:
            break
    N = len(manifest)
    if N == 0:
        raise SystemExit('No sequences found — check --meta / --datasets.')
    print(f'Manifest: {N} sequences across {len(ds_by_key)} preprocessing config(s)')

    # ---- gating logits for all sequences (vectorized, one pass) ----
    gating_logits = None
    has_gating = np.zeros(N, bool)
    image_keys = [''] * N
    if args.clustering_dir is not None:
        from build_unique_image_jsonl import image_key
        cdir = Path(args.clustering_dir)
        centroids = np.load(cdir / f'{args.prefix}_centroids.npy').astype(np.float64)
        cnorm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
        mean_file = cdir / f'{args.prefix}_global_mean.npy'
        gmean = np.load(mean_file).astype(np.float64) if mean_file.exists() else None
        feats = np.load(args.features_prefix + '.npy')
        keys = json.load(open(args.features_prefix + '_keys.json'))
        key_to_idx = {k: i for i, k in enumerate(keys)}
        K_clu = cnorm.shape[0]
        gating_logits = np.full((N, K_clu), np.nan, np.float32)
        fidx = np.full(N, -1, np.int64)
        for i, e in enumerate(manifest):
            if e['image']:
                k = image_key(e['image'].strip().replace('\\', '/'))
                image_keys[i] = k
                j = key_to_idx.get(k)
                if j is not None:
                    fidx[i] = j
        have = fidx >= 0
        Fh = feats[fidx[have]].astype(np.float64)
        if gmean is not None:
            Fh = Fh - gmean
        Fh = Fh / np.linalg.norm(Fh, axis=1, keepdims=True).clip(1e-8)
        gating_logits[have] = (Fh @ cnorm.T).astype(np.float32)
        has_gating = have
        print(f'Gating: {K_clu} centroids, mean-subtracted={gmean is not None}, '
              f'{int(have.sum())}/{N} sequences have features')

    # ---- result arrays (+ resume) ----
    loglik = np.full((N, K_exp), np.nan, np.float64)
    n_resp = np.full(N, -1, np.int32)
    done = np.zeros(N, bool)
    partial = Path(str(args.output) + '.partial.npz')
    if args.resume and partial.exists():
        pd = np.load(partial)
        if pd['loglik'].shape == loglik.shape:
            loglik, n_resp, done = pd['loglik'], pd['n_resp_tokens'], pd['done']
            print(f'Resumed: {int(done.sum())}/{N} already done')

    # ---- process order: only not-done, sorted by (proxy) length ----
    est = np.array([256 + len(m['line']) // 3 for m in manifest], np.int64)
    todo = [i for i in range(N) if not done[i]]
    todo.sort(key=lambda i: est[i])
    print(f'To evaluate: {len(todo)} sequences')

    def save_partial():
        tmp = str(partial) + '.tmp.npz'
        np.savez(tmp, loglik=loglik, n_resp_tokens=n_resp, done=done)
        os.replace(tmp, partial)

    if todo:
        mds = ManifestDataset(manifest, todo, ds_by_key)
        sampler = TokenBudgetSampler(list(range(len(todo))), est[todo],
                                     args.max_batch_tokens, args.max_batch_size)
        loader = DataLoader(mds, batch_sampler=sampler, collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=False, persistent_workers=args.num_workers > 0)

        processed, t0 = 0, time.time()
        for batch, gidxs, failed in loader:
            for g in failed:
                done[g] = True  # give up on unloadable lines (loglik stays NaN)
            if batch is not None:
                nr = n_resp_of(batch)
                lls = np.stack([batched_loglik(m, batch, device) for m in experts], axis=1)
                for row, g in enumerate(gidxs):
                    loglik[g] = lls[row]
                    n_resp[g] = nr[row]
                    done[g] = True
            processed += len(gidxs) + len(failed)
            if processed % max(args.checkpoint_every, 1) < (len(gidxs) + len(failed)):
                rate = processed / max(time.time() - t0, 1e-6)
                eta = (len(todo) - processed) / max(rate, 1e-6)
                print(f'  {processed}/{len(todo)}  {rate:.1f} seq/s  ETA {eta/60:.1f} min',
                      flush=True)
                save_partial()
        save_partial()

    # ---- final outputs ----
    out = dict(loglik=loglik, n_resp_tokens=n_resp,
               repeat_time=np.array([m['repeat_time'] for m in manifest], np.float32),
               done=done)
    if gating_logits is not None:
        out['gating_logits'] = gating_logits
        out['has_gating'] = has_gating
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outpath, **out)
    Path(str(outpath) + '.json').write_text(json.dumps(dict(
        meta=str(args.meta), expert_checkpoints=list(args.expert_checkpoints),
        clustering_dir=str(args.clustering_dir) if args.clustering_dir else None,
        n_sequences=N, n_experts=K_exp, n_done=int(done.sum()),
        datasets=[m['ds_name'] for m in manifest],
        image_paths=[m['image'] for m in manifest],
        image_keys=image_keys if gating_logits is not None else None), indent=2))

    ok = done & np.isfinite(loglik).all(1)
    tok = int(n_resp[ok].sum())
    print(f'\nDone {int(done.sum())}/{N} ({int(ok.sum())} with finite loglik) -> {outpath}')
    if tok:
        print(f'  mean per-token logp (expert 0): {loglik[ok, 0].sum() / tok:.4f}')
    if gating_logits is not None:
        print(f'  sequences with gating: {int(has_gating.sum())}/{N}')


if __name__ == '__main__':
    main()
