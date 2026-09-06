#!/usr/bin/env python3
"""Train the tiny VLM from scratch on the synthetic shapes dataset.

Single GPU:
    python -m synthetic_data.vlm.train --out-dir synthetic_data/vlm/runs/base

Two GPUs (DistributedDataParallel; --batch-size is *per device*):
    torchrun --nproc_per_node=2 -m synthetic_data.vlm.train \
        --out-dir synthetic_data/vlm/runs/dense --batch-size 64

Evaluation is greedy decoding scored by exact match, broken down by task and by
the generative parameter ``alpha`` -- the split the dataset was built to expose.

Two baselines are reported so the numbers can be read:
  * ``--no-vision``: same model, visual features zeroed. Anything it reaches is
    obtainable from the question alone (answer priors), not from the image.
  * a majority-answer-per-task baseline, computed from the training split.
"""

import io
import os
import json
import time
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import (CharTokenizer, VQADataset, FeatureStats, collate,
                   load_records, split_by_image, BOS_ID, SEP_ID)
from .model import TinyVLM


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

TASK_NAME = {1: 'ocr', 2: 'count_by_type', 3: 'count_by_color',
             4: 'name_types', 5: 'count_by_angles'}


def char_error_rate(pred, gold):
    """Levenshtein distance normalised by the gold length."""
    if not gold:
        return 0.0 if not pred else 1.0
    prev = list(range(len(pred) + 1))
    for j, g in enumerate(gold, 1):
        cur = [j]
        for i, p in enumerate(pred, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1,
                           prev[i - 1] + (p != g)))
        prev = cur
    return prev[-1] / len(gold)


def set_f1(pred, gold):
    p = {t.strip() for t in pred.split(',') if t.strip()}
    g = {t.strip() for t in gold.split(',') if t.strip()}
    if not p and not g:
        return 1.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec)


def score(records, preds):
    """Aggregate metrics overall, per task, and per alpha bin."""
    overall = {'n': 0, 'em': 0}
    per_task = defaultdict(lambda: {'n': 0, 'em': 0, 'cer': 0.0,
                                    'mae': 0.0, 'mae_n': 0, 'f1': 0.0})
    per_alpha = defaultdict(lambda: {'n': 0, 'em': 0, 'ocr_n': 0, 'ocr_em': 0,
                                     'shape_n': 0, 'shape_em': 0})
    for r, pred in zip(records, preds):
        gold = r['answer']
        em = int(pred == gold)
        overall['n'] += 1
        overall['em'] += em

        t = per_task[r['task_id']]
        t['n'] += 1
        t['em'] += em
        if r['task_id'] == 1:
            t['cer'] += char_error_rate(pred, gold)
        elif r['task_id'] == 4:
            t['f1'] += set_f1(pred, gold)
        else:
            try:
                t['mae'] += abs(int(pred) - int(gold))
                t['mae_n'] += 1
            except ValueError:
                pass

        b = min(int(r['alpha'] * 5), 4)
        a = per_alpha[b]
        a['n'] += 1
        a['em'] += em
        if r['task_type'] == 'ocr':
            a['ocr_n'] += 1
            a['ocr_em'] += em
        else:
            a['shape_n'] += 1
            a['shape_em'] += em

    out = {'exact_match': overall['em'] / max(1, overall['n']),
           'n': overall['n'], 'per_task': {}, 'per_alpha_bin': {}}
    for tid, d in sorted(per_task.items()):
        e = {'name': TASK_NAME[tid], 'n': d['n'], 'exact_match': d['em'] / d['n']}
        if tid == 1:
            e['cer'] = d['cer'] / d['n']
        elif tid == 4:
            e['set_f1'] = d['f1'] / d['n']
        else:
            e['mae'] = d['mae'] / d['mae_n'] if d['mae_n'] else None
            e['unparseable'] = d['n'] - d['mae_n']
        out['per_task'][tid] = e
    for b, d in sorted(per_alpha.items()):
        out['per_alpha_bin'][b] = {
            'range': [b / 5, (b + 1) / 5], 'n': d['n'],
            'exact_match': d['em'] / d['n'],
            'ocr_em': d['ocr_em'] / d['ocr_n'] if d['ocr_n'] else None,
            'ocr_n': d['ocr_n'],
            'shape_em': d['shape_em'] / d['shape_n'] if d['shape_n'] else None,
            'shape_n': d['shape_n'],
        }
    return out


def majority_baseline(train_records, eval_records):
    """Per-task most frequent training answer; the floor any model must beat."""
    by_task = defaultdict(Counter)
    for r in train_records:
        by_task[r['task_id']][r['answer']] += 1
    guess = {t: c.most_common(1)[0][0] for t, c in by_task.items()}
    return [guess.get(r['task_id'], '') for r in eval_records]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(model, dataset, device, batch_size=256, max_new_tokens=64):
    """Greedy decode every record. Prompts are bucketed by length so no padding
    is needed (question templates are few, so lengths repeat heavily)."""
    model.eval()
    buckets = defaultdict(list)
    for i in range(len(dataset)):
        buckets[len(dataset.prompt_ids(dataset.records[i]))].append(i)

    preds = [None] * len(dataset)
    for _, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            items = [dataset[i] for i in chunk]
            feats = torch.stack([it['feat'] for it in items]).to(device)
            prompts = torch.tensor(
                [dataset.prompt_ids(dataset.records[i]) for i in chunk],
                dtype=torch.long, device=device)
            outs = model.generate(feats, prompts, max_new_tokens=max_new_tokens)
            for i, ids in zip(chunk, outs):
                preds[i] = dataset.tok.decode(ids)
    return preds


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        feat = batch['feat'].to(device, non_blocking=True)
        ids = batch['ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        ntok = int((labels[:, 1:] != -100).sum())
        _, loss = model(feat, ids, labels)
        tot += loss.item() * ntok
        n += ntok
    return tot / max(1, n)


def fmt_report(name, rep):
    lines = [f'{name}: exact match {100 * rep["exact_match"]:.2f}% '
             f'(n={rep["n"]})']
    for tid, e in rep['per_task'].items():
        extra = ''
        if 'cer' in e:
            extra = f'  CER {e["cer"]:.3f}'
        elif 'set_f1' in e:
            extra = f'  set-F1 {e["set_f1"]:.3f}'
        elif e.get('mae') is not None:
            extra = f'  MAE {e["mae"]:.3f}  unparseable {e["unparseable"]}'
        lines.append(f'  task {tid} {e["name"]:<16} n={e["n"]:<5} '
                     f'EM {100 * e["exact_match"]:6.2f}%{extra}')
    lines.append('  by alpha bin:')
    for b, e in rep['per_alpha_bin'].items():
        o = f'{100 * e["ocr_em"]:.1f}%' if e['ocr_em'] is not None else '  -  '
        sh = f'{100 * e["shape_em"]:.1f}%' if e['shape_em'] is not None else '  -  '
        lines.append(f'    alpha [{e["range"][0]:.1f},{e["range"][1]:.1f}) '
                     f'n={e["n"]:<5} EM {100 * e["exact_match"]:6.2f}%   '
                     f'ocr {o} (n={e["ocr_n"]})   shape {sh} (n={e["shape_n"]})')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def ddp_setup():
    """Initialise from torchrun's env vars; returns (rank, world_size, local_rank)."""
    if 'RANK' not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    return rank, world, local


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--keys', default=None)
    ap.add_argument('--out-dir', default='synthetic_data/vlm/runs/base')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--max-train', type=int, default=0,
                    help='keep only training images with image_id < MAX_TRAIN. The '
                         'generator numbers train images first, so this yields nested '
                         'subsets and leaves val/test untouched -- and an expert shard '
                         'inherits exactly the same image subset.')
    ap.add_argument('--tokenizer', default=None,
                    help='reuse an existing tokenizer.json instead of building the '
                         'vocabulary from this file. Required when several models '
                         'must share an index space (e.g. token-wise ensembling).')
    ap.add_argument('--feature-norm', choices=('standardize', 'l2', 'none'),
                    default='standardize')
    ap.add_argument('--no-vision', action='store_true',
                    help='zero the CLIP features (blind ablation)')
    # model
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-prefix', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--moe-experts', type=int, default=1,
                    help='FFN experts per block (1 = dense). Top-1 routing, so the '
                         'active parameter count is unchanged.')
    ap.add_argument('--moe-aux-coef', type=float, default=0.01,
                    help='weight on the Switch load-balancing auxiliary loss')
    # optimisation
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-frac', type=float, default=0.03)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=5,
                    help='epochs between full greedy-decode evaluations')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rank, world, local_rank = ddp_setup()
    is_main = rank == 0
    distributed = world > 1

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{local_rank}') if distributed \
        else torch.device(args.device)
    if distributed:
        log(f'DDP: {world} processes, per-device batch {args.batch_size} '
            f'(effective {world * args.batch_size})')

    records, feats = load_records(args.annotations, args.features, args.keys)
    splits = split_by_image(records, args.val_frac, args.test_frac, args.split_seed)
    if args.max_train:
        splits['train'] = [r for r in splits['train']
                           if r['image_id'] < args.max_train]
    log(f'{len(records)} QA pairs over {len({r["image_id"] for r in records})} images')
    for k in ('train', 'val', 'test'):
        imgs = len({r['image_id'] for r in splits[k]})
        log(f'  {k:<6} {len(splits[k]):>6} pairs / {imgs:>5} images')

    tok = (CharTokenizer.load(args.tokenizer) if args.tokenizer else
           CharTokenizer.from_texts([r['question'] for r in records]
                                    + [r['answer'] for r in records]))
    if is_main:
        tok.save(out_dir / 'tokenizer.json')
    stats = FeatureStats.fit(feats, [r['row'] for r in splits['train']],
                             args.feature_norm)

    ds = {k: VQADataset(splits[k], feats, tok, stats, no_vision=args.no_vision)
          for k in splits}
    # Size the positional table from every split, not just train: with a small
    # --max-train the longest training sequence can be shorter than the longest
    # val/test one, and evaluation would then overflow the table.
    max_len = max(len(ds[k][i]['ids'])
                  for k in ('train', 'val', 'test')
                  for i in range(len(ds[k]))) + 8
    log(f'vocab {len(tok)} symbols, max text length {max_len}')

    train_sampler = DistributedSampler(ds['train'], num_replicas=world, rank=rank,
                                       shuffle=True, drop_last=True) \
        if distributed else None
    train_loader = DataLoader(ds['train'], batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(ds['val'], batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    model = TinyVLM(len(tok), feat_dim=feats.shape[1], d_model=args.d_model,
                    n_layers=args.n_layers, n_heads=args.n_heads,
                    n_prefix=args.n_prefix, max_len=max_len,
                    dropout=args.dropout, n_experts=args.moe_experts).to(device)
    if args.moe_experts > 1:
        log(f'model: MoE top-1, {args.moe_experts} FFN experts/block  '
            f'total {model.n_params() / 1e6:.2f}M  '
            f'active {model.n_active_params() / 1e6:.2f}M  '
            f'aux coef {args.moe_aux_coef}')
    else:
        log(f'model: {model.n_params() / 1e6:.2f}M trainable parameters'
            + ('  [NO-VISION ABLATION]' if args.no_vision else ''))
    core = model
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    decay, no_decay = [], []
    for n, p in core.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': args.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    amp = device.type == 'cuda'
    history, best = [], {'val_em': -1.0, 'epoch': -1}
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        run, ntok = 0.0, 0
        aux_track = [] if args.moe_experts > 1 else None
        for batch in train_loader:
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            feat = batch['feat'].to(device, non_blocking=True)
            ids = batch['ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                _, loss = model(feat, ids, labels)
                ce = loss.detach()
                aux = core.aux_loss()
                if aux is not None:
                    loss = loss + args.moe_aux_coef * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            opt.step()
            n = int((labels[:, 1:] != -100).sum())
            run += ce.item() * n      # report the CE, not CE + aux
            ntok += n
            if aux_track is not None and aux is not None:
                aux_track.append(float(aux.detach()))
            step += 1

        if distributed:
            agg = torch.tensor([run, float(ntok)], device=device)
            dist.all_reduce(agg)
            run, ntok = agg[0].item(), int(agg[1].item())

        entry = {'epoch': epoch, 'train_loss': run / max(1, ntok)}
        if aux_track:
            entry['aux_loss'] = sum(aux_track) / len(aux_track)
            fr = core.expert_fractions()
            if fr:
                entry['expert_fractions'] = [
                    [round(float(x), 4) for x in f.tolist()] for f in fr]
        if not is_main:
            continue
        is_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if is_eval:
            entry['val_loss'] = eval_loss(core, val_loader, device)
        if is_eval:
            rep = score(splits['val'], predict(core, ds['val'], device))
            entry['val_exact_match'] = rep['exact_match']
            extra = (f'  aux {entry["aux_loss"]:.4f}'
                     if 'aux_loss' in entry else '')
            print(f'epoch {epoch:>3}  train {entry["train_loss"]:.4f}  '
                  f'val {entry["val_loss"]:.4f}  '
                  f'val EM {100 * rep["exact_match"]:.2f}%{extra}  '
                  f'({time.time() - t0:.0f}s)')
            if rep['exact_match'] > best['val_em']:
                best = {'val_em': rep['exact_match'], 'epoch': epoch}
                torch.save({'model': core.state_dict(),
                            'args': vars(args),
                            'stats': stats.state_dict(),
                            'max_len': max_len,
                            'n_experts': args.moe_experts,
                            'vocab': tok.itos},
                           out_dir / 'best.pt')
        else:
            print(f'epoch {epoch:>3}  train {entry["train_loss"]:.4f}  '
                  f'({time.time() - t0:.0f}s)')
        history.append(entry)

    if distributed:
        dist.barrier()
    if not is_main:
        dist.destroy_process_group()
        return

    # --- final test evaluation with the best checkpoint --------------------- #
    ckpt_path = out_dir / 'best.pt'
    if ckpt_path.exists():
        core.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        print(f'\nloaded best checkpoint (epoch {best["epoch"]}, '
              f'val EM {100 * best["val_em"]:.2f}%)')

    results = {'args': vars(args), 'history': history, 'best': best}
    print()
    for split in ('val', 'test'):
        rep = score(splits[split], predict(core, ds[split], device))
        results[split] = rep
        print(fmt_report(split.upper(), rep))
        print()
    maj = score(splits['test'], majority_baseline(splits['train'], splits['test']))
    results['test_majority_baseline'] = maj
    print(fmt_report('TEST majority-answer baseline', maj))

    # A few qualitative examples.
    preds = predict(core, ds['test'], device)
    samples = [{'image_id': r['image_id'], 'alpha': r['alpha'],
                'task_id': r['task_id'], 'question': r['question'],
                'gold': r['answer'], 'pred': p}
               for r, p in list(zip(splits['test'], preds))[:25]]
    results['examples'] = samples
    print('\nexamples (test):')
    for s in samples[:12]:
        mark = 'OK ' if s['pred'] == s['gold'] else '   '
        print(f'  {mark} a={s["alpha"]:.2f} t{s["task_id"]} '
              f'gold={s["gold"]!r:<28} pred={s["pred"]!r}')

    results['world_size'] = world
    results['effective_batch_size'] = world * args.batch_size
    (out_dir / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out_dir / "results.json"} and {ckpt_path}')
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
