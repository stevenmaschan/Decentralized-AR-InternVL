#!/usr/bin/env python3
"""Evaluate the MoE model and inspect what its learned router actually does.

Beyond the usual accuracy report this asks the question the hand-made splits were
probing: given no supervision about tasks or alpha, does a learned top-1 router
rediscover the OCR / shape-reasoning division that context splitting exploits?

For every layer it reports the share of tokens sent to expert 0, broken down by
task family and by alpha bin. A router that ignores the input sends ~50% either
way in every group; one that specialises separates the groups.

    python -m synthetic_data.vlm.eval_moe --run synthetic_data/vlm/runs/moe_20k
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (CharTokenizer, FeatureStats, VQADataset, collate,
                   load_records, get_split, PAD_ID)
from .model import TinyVLM
from .train import predict, score, fmt_report


def load_run(run_dir, feat_dim, device):
    ck = torch.load(Path(run_dir) / 'best.pt', map_location=device)
    tok = CharTokenizer(ck['vocab'][4:])
    a = ck['args']
    m = TinyVLM(len(tok), feat_dim, a['d_model'], a['n_layers'], a['n_heads'],
                a['n_prefix'], ck['max_len'], 0.0,
                n_experts=ck.get('n_experts', 1)).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, tok, FeatureStats.from_state(ck['stats'])


@torch.no_grad()
def routing_stats(model, ds, records, device, batch_size=256):
    """Share of tokens routed to expert 0, per layer, grouped by task and alpha."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    n_layers = len(model.blocks)
    groups = defaultdict(lambda: np.zeros((n_layers, 2)))   # [to_e0, total]
    pos = 0
    for batch in loader:
        feat = batch['feat'].to(device)
        ids = batch['ids'].to(device)
        model.backbone(feat, ids)
        valid = torch.cat([torch.ones(ids.size(0), model.n_prefix,
                                      dtype=torch.bool, device=device),
                           ids != PAD_ID], dim=1)
        tops = [b.mlp.last_top for b in model.blocks]
        for j in range(ids.size(0)):
            r = records[pos + j]
            keys = [('all', ''), ('task', r['task_type']),
                    ('alpha', f'{min(int(r["alpha"] * 5), 4) / 5:.1f}-'
                              f'{(min(int(r["alpha"] * 5), 4) + 1) / 5:.1f}')]
            v = valid[j]
            for L, top in enumerate(tops):
                t = top[j][v]
                e0 = int((t == 0).sum())
                for g, k in keys:
                    groups[(g, k)][L] += (e0, t.numel())
        pos += ids.size(0)
    return {k: (v[:, 0] / np.maximum(1, v[:, 1])) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--run', default='synthetic_data/vlm/runs/moe_20k')
    ap.add_argument('--dense', default='synthetic_data/vlm/runs/sweep/size__20k__dense')
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    records, feats = load_records(args.annotations, args.features)
    ev = get_split(records, args.split)
    print(f'{args.split}: {len(ev)} QA pairs')

    out = {}
    m, tok, st = load_run(args.run, feats.shape[1], device)
    print(f'MoE: {m.n_experts} experts, total {m.n_params()/1e6:.2f}M, '
          f'active {m.n_active_params()/1e6:.2f}M')
    ds = VQADataset(ev, feats, tok, st)
    out['moe'] = score(ev, predict(m, ds, device))
    print('\n' + fmt_report('MoE', out['moe']))

    if args.dense and Path(args.dense, 'best.pt').exists():
        dm, dtok, dst = load_run(args.dense, feats.shape[1], device)
        out['dense'] = score(ev, predict(dm, VQADataset(ev, feats, dtok, dst), device))
        print('\n' + fmt_report('DENSE', out['dense']))
        d, mo = 100 * out['dense']['exact_match'], 100 * out['moe']['exact_match']
        print(f'\ndense {d:.2f}%   MoE {mo:.2f}%   delta {mo - d:+.2f}')

    if m.n_experts > 1:
        stats = routing_stats(m, ds, ev, device)
        n_layers = len(m.blocks)
        print('\nshare of tokens routed to expert 0 '
              '(50% = router ignores the input)')
        header = 'group'.ljust(22) + ''.join(f'L{i}'.rjust(8) for i in range(n_layers))
        print(header)
        print('-' * len(header))
        order = [('all', ''), ('task', 'ocr'), ('task', 'shape_reasoning')]
        order += sorted(k for k in stats if k[0] == 'alpha')
        for k in order:
            if k not in stats:
                continue
            label = k[1] or 'all tokens'
            print(f'{label:<22}' + ''.join(f'{100*x:>7.1f}%' for x in stats[k]))
        sep = np.abs(stats[('task', 'ocr')] - stats[('task', 'shape_reasoning')])
        print(f'\nOCR vs shape separation per layer: '
              + '  '.join(f'{100*x:.1f}pp' for x in sep))
        print(f'max separation {100*sep.max():.1f}pp '
              f'(0 = router is blind to task, 100 = fully specialised)')
        out['routing'] = {f'{g}:{k}': [float(x) for x in v]
                          for (g, k), v in stats.items()}

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
