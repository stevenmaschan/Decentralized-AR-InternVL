#!/usr/bin/env python3
"""Train the full-model MoE (2 whole experts + a learned sample-level router).

    python -m synthetic_data.vlm.train_moe_full --out-dir .../moefull_100k \
        --annotations ... --features ... --epochs 50 --batch-size 128

Reuses the data plumbing and metrics from train.py; only the model and the loss
differ. Reported "train CE" is the *routed* cross-entropy -- the loss the argmax
expert would pay at inference -- not the mixture NLL that is optimised, so the
curves are comparable with the dense runs.
"""

import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (CharTokenizer, VQADataset, FeatureStats, collate,
                   load_records, get_split, IGNORE, PAD_ID)
from .moe_full import FullModelMoE
from .train import predict, score, fmt_report


@torch.no_grad()
def routed_predict(model, ds, records, device, batch_size=256):
    """Route each sample with the router argmax, then greedy-decode with that expert."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    assign = np.zeros(len(records), dtype=int)
    pos = 0
    for b in loader:
        feat, ids = b['feat'].to(device), b['ids'].to(device)
        labels = b['labels'].to(device)
        pm = (labels == IGNORE) & (ids != PAD_ID)
        a = model.route(feat, ids, pm).cpu().numpy()
        assign[pos:pos + len(a)] = a
        pos += len(a)

    preds = [None] * len(records)
    for e in range(model.n_experts):
        idx = np.where(assign == e)[0]
        if not len(idx):
            continue
        sub = [records[i] for i in idx]
        sub_ds = VQADataset(sub, ds.feats, ds.tok, ds.stats)
        for i, p in zip(idx, predict(model.experts[e], sub_ds, device)):
            preds[i] = p
    return preds, assign


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-prefix', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--n-experts', type=int, default=2)
    ap.add_argument('--routing', choices=('sparse', 'soft'), default='sparse',
                    help="'sparse': hard top-1, one expert per sample, matching "
                         "inference (Switch style); 'soft': exact mixture "
                         "likelihood over both experts (Jacobs style)")
    ap.add_argument('--jitter', type=float, default=0.01,
                    help='multiplicative router-input jitter for exploration')
    ap.add_argument('--aux-coef', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-frac', type=float, default=0.03)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    records, feats = load_records(args.annotations, args.features)
    splits = {k: get_split(records, k) for k in ('train', 'val', 'test')}
    if args.max_train:
        splits['train'] = [r for r in splits['train']
                           if r['image_id'] < args.max_train]
    print(f'train {len(splits["train"])}  val {len(splits["val"])}  '
          f'test {len(splits["test"])}')

    tok = CharTokenizer.from_texts([r['question'] for r in records]
                                   + [r['answer'] for r in records])
    tok.save(out_dir / 'tokenizer.json')
    stats = FeatureStats.fit(feats, [r['row'] for r in splits['train']])
    ds = {k: VQADataset(v, feats, tok, stats) for k, v in splits.items()}
    max_len = max(len(ds[k][i]['ids']) for k in ds for i in range(len(ds[k]))) + 8

    model = FullModelMoE(len(tok), feat_dim=feats.shape[1],
                         n_experts=args.n_experts, routing=args.routing,
                         jitter=args.jitter,
                         d_model=args.d_model,
                         n_layers=args.n_layers, n_heads=args.n_heads,
                         n_prefix=args.n_prefix, max_len=max_len,
                         dropout=args.dropout).to(device)
    print(f'full-model MoE ({args.routing} top-1): {args.n_experts} experts, '
          f'total {model.n_params()/1e6:.2f}M, active {model.n_active_params()/1e6:.2f}M')

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    loader = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    total = max(1, len(loader)) * args.epochs
    warmup = max(1, int(args.warmup_frac * total))

    def lr_at(s):
        if s < warmup:
            return args.lr * s / warmup
        p = (s - warmup) / max(1, total - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    best = {'val_em': -1.0, 'epoch': -1}
    history, step, t0 = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        mix_s = ce_s = n_s = 0.0
        fr = None
        for b in loader:
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            feat, ids = b['feat'].to(device), b['ids'].to(device)
            labels = b['labels'].to(device)
            loss = model(feat, ids, labels)
            total_loss = loss + args.aux_coef * model.last_aux
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            k = int((labels[:, 1:] != IGNORE).sum())
            mix_s += model.mixture_nll.item() * k
            ce_s += model.routed_ce.item() * k
            n_s += k
            fr = model.last_frac
            step += 1

        e = {'epoch': epoch, 'mixture_nll': mix_s / n_s, 'routed_ce': ce_s / n_s,
             'expert_fraction': [round(float(x), 4) for x in fr.tolist()]}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            preds, assign = routed_predict(model, ds['val'], splits['val'], device)
            rep = score(splits['val'], preds)
            e['val_exact_match'] = rep['exact_match']
            print(f'epoch {epoch:>3}  mixNLL {e["mixture_nll"]:.4f}  '
                  f'routedCE {e["routed_ce"]:.4f}  val EM '
                  f'{100*rep["exact_match"]:.2f}%  split {e["expert_fraction"]}  '
                  f'({time.time()-t0:.0f}s)')
            if rep['exact_match'] > best['val_em']:
                best = {'val_em': rep['exact_match'], 'epoch': epoch}
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'stats': stats.state_dict(), 'max_len': max_len,
                            'vocab': tok.itos}, out_dir / 'best.pt')
        else:
            print(f'epoch {epoch:>3}  mixNLL {e["mixture_nll"]:.4f}  '
                  f'routedCE {e["routed_ce"]:.4f}  split {e["expert_fraction"]}  '
                  f'({time.time()-t0:.0f}s)')
        history.append(e)

    ck = out_dir / 'best.pt'
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)['model'])
        print(f'\nloaded best (epoch {best["epoch"]}, '
              f'val EM {100*best["val_em"]:.2f}%)')
    res = {'args': vars(args), 'history': history, 'best': best}
    for sp in ('val', 'test'):
        preds, assign = routed_predict(model, ds[sp], splits[sp], device)
        res[sp] = score(splits[sp], preds)
        print('\n' + fmt_report(sp.upper(), res[sp]))
        by = defaultdict(lambda: [0, 0])
        for r, a in zip(splits[sp], assign):
            by[r['task_type']][a] += 1
        print('  router assignment by task family:')
        for k, v in by.items():
            tot = sum(v)
            print(f'    {k:<18} expert0 {100*v[0]/tot:5.1f}%   '
                  f'expert1 {100*v[1]/tot:5.1f}%   (n={tot})')
        res[sp + '_routing'] = {k: v for k, v in by.items()}
    (out_dir / 'results.json').write_text(json.dumps(res, indent=2))
    print(f'\nWrote {out_dir / "results.json"}')


if __name__ == '__main__':
    main()
