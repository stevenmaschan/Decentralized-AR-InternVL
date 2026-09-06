#!/usr/bin/env python3
"""Compare the dense model against the two alpha experts under oracle routing.

Each test sample is answered by the expert that owns its alpha range
(alpha < threshold -> lo, else hi), and the combined predictions are scored
against the dense model on exactly the same test set.

    python -m synthetic_data.vlm.eval_routed
"""

import json
import argparse
from pathlib import Path

import torch

from .data import (CharTokenizer, FeatureStats, VQADataset, load_records,
                   get_split)
from .model import TinyVLM
from .train import predict, score, fmt_report


def load_run(run_dir, feat_dim, device):
    ck = torch.load(Path(run_dir) / 'best.pt', map_location=device)
    tok = CharTokenizer(ck['vocab'][4:])
    a = ck['args']
    m = TinyVLM(len(tok), feat_dim, a['d_model'], a['n_layers'], a['n_heads'],
                a['n_prefix'], ck['max_len'], 0.0).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, tok, FeatureStats.from_state(ck['stats'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--dense', default='synthetic_data/vlm/runs/dense')
    ap.add_argument('--expert-lo', default='synthetic_data/vlm/runs/expert_lo')
    ap.add_argument('--expert-hi', default='synthetic_data/vlm/runs/expert_hi')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--route-by', choices=('alpha', 'task'), default='alpha',
                    help="'alpha': image-based routing (alpha < threshold -> lo); "
                         "'task': context routing, OCR questions -> first expert")
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', default='synthetic_data/vlm/runs/routed_results.json')
    args = ap.parse_args()

    device = torch.device(args.device)
    records, feats = load_records(args.annotations, args.features)
    test = get_split(records, args.split)
    print(f'{args.split} set: {len(test)} QA pairs')

    out = {}

    dense, dtok, dstats = load_run(args.dense, feats.shape[1], device)
    dense_pred = predict(dense, VQADataset(test, feats, dtok, dstats), device)
    out['dense'] = score(test, dense_pred)
    print('\n' + fmt_report('DENSE', out['dense']))

    # Routing: every sample answered by the expert that owns it. For 'alpha' the
    # owner is decided by the image, for 'task' by the question type.
    if args.route_by == 'alpha':
        rules = ((args.expert_lo, lambda r: r['alpha'] < args.threshold),
                 (args.expert_hi, lambda r: r['alpha'] >= args.threshold))
    else:
        rules = ((args.expert_lo, lambda r: r['task_type'] == 'ocr'),
                 (args.expert_hi, lambda r: r['task_type'] != 'ocr'))
    routed = [None] * len(test)
    for run_dir, keep in rules:
        idx = [i for i, r in enumerate(test) if keep(r)]
        m, tok, st = load_run(run_dir, feats.shape[1], device)
        sub = [test[i] for i in idx]
        preds = predict(m, VQADataset(sub, feats, tok, st), device)
        rep = score(sub, preds)
        name = Path(run_dir).name
        out[name] = rep
        print('\n' + fmt_report(name.upper() + ' (own shard only)', rep))
        for i, p in zip(idx, preds):
            routed[i] = p

    assert all(p is not None for p in routed)
    out['routed'] = score(test, routed)
    out['route_by'] = args.route_by
    label = ('ROUTED (oracle alpha routing)' if args.route_by == 'alpha'
             else 'ROUTED (context / question-type routing)')
    print('\n' + fmt_report(label, out['routed']))

    d, r = out['dense']['exact_match'], out['routed']['exact_match']
    print(f'\ndense  {100 * d:.2f}%   routed {100 * r:.2f}%   '
          f'delta {100 * (r - d):+.2f} points')
    agree = sum(a == b for a, b in zip(dense_pred, routed)) / len(test)
    print(f'dense and routed give the same answer on {100 * agree:.1f}% of samples')

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
