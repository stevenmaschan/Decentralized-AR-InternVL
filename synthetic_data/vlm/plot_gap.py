#!/usr/bin/env python3
"""Fine-grained histogram of dense-minus-routed accuracy against alpha.

Re-decodes both systems on the evaluation split so the gap can be binned at any
resolution, rather than at the 5 bins eval_routed records. Because both systems
answer the *same* samples, the difference is paired: the error bar on each bin is
the standard error of the per-sample difference d_i = correct_dense -
correct_routed, which is much tighter than treating the two accuracies as
independent.

Positive bars = dense better; negative = routing better.

    python -m synthetic_data.vlm.plot_gap --bins 20 \
        --dense synthetic_data/vlm/runs/sweep/size__20k__dense \
        --expert-lo synthetic_data/vlm/runs/sweep/size__20k__expert_lo \
        --expert-hi synthetic_data/vlm/runs/sweep/size__20k__expert_hi
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from .data import CharTokenizer, FeatureStats, VQADataset, load_records, split_by_image
from .model import TinyVLM
from .train import predict


def load_run(run_dir, feat_dim, device):
    ck = torch.load(Path(run_dir) / 'best.pt', map_location=device)
    tok = CharTokenizer(ck['vocab'][4:])
    a = ck['args']
    m = TinyVLM(len(tok), feat_dim, a['d_model'], a['n_layers'], a['n_heads'],
                a['n_prefix'], ck['max_len'], 0.0).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, tok, FeatureStats.from_state(ck['stats'])


def binned_gap(alpha, diff, edges):
    """Mean paired difference and its standard error, per bin."""
    idx = np.clip(np.digitize(alpha, edges) - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            out.append((b, np.nan, np.nan, 0))
            continue
        d = diff[m]
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out.append((b, 100 * d.mean(), 100 * se, n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--dense', default='synthetic_data/vlm/runs/sweep/size__20k__dense')
    ap.add_argument('--expert-lo',
                    default='synthetic_data/vlm/runs/sweep/size__20k__expert_lo')
    ap.add_argument('--expert-hi',
                    default='synthetic_data/vlm/runs/sweep/size__20k__expert_hi')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--route-by', choices=('alpha', 'task', 'mixture'),
                    default='alpha',
                    help="'alpha': --expert-lo owns alpha<threshold; "
                         "'task': --expert-lo owns OCR questions; "
                         "'mixture': no routing -- both experts answer every "
                         "sample and their next-token distributions are averaged "
                         "0.5/0.5 (for a partition that carries no signal)")
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--title', default='20k train')
    ap.add_argument('--output',
                    default='synthetic_data/vlm/runs/sweep/gap_20k.png')
    args = ap.parse_args()

    device = torch.device(args.device)
    records, feats = load_records(args.annotations, args.features)
    ev = split_by_image(records)[args.split]
    print(f'{args.split}: {len(ev)} QA pairs, {args.bins} bins')

    dm, dtok, dst = load_run(args.dense, feats.shape[1], device)
    dense_pred = predict(dm, VQADataset(ev, feats, dtok, dst), device)

    if args.route_by == 'mixture':
        from .eval_mixture import mixture_predict
        members = [load_run(d, feats.shape[1], device)
                   for d in (args.expert_lo, args.expert_hi)]
        if members[0][1].itos != members[1][1].itos:
            raise SystemExit('mixture members must share a vocabulary')
        routed = mixture_predict(members, [0.5, 0.5], ev, feats, device)
    else:
        if args.route_by == 'alpha':
            rules = ((args.expert_lo, lambda r: r['alpha'] < args.threshold),
                     (args.expert_hi, lambda r: r['alpha'] >= args.threshold))
        else:
            rules = ((args.expert_lo, lambda r: r['task_type'] == 'ocr'),
                     (args.expert_hi, lambda r: r['task_type'] != 'ocr'))
        routed = [None] * len(ev)
        for run_dir, keep in rules:
            idx = [i for i, r in enumerate(ev) if keep(r)]
            m, tok, st = load_run(run_dir, feats.shape[1], device)
            sub = [ev[i] for i in idx]
            for i, p in zip(idx, predict(m, VQADataset(sub, feats, tok, st), device)):
                routed[i] = p
    assert all(p is not None for p in routed)

    alpha = np.array([r['alpha'] for r in ev])
    is_ocr = np.array([r['task_type'] == 'ocr' for r in ev])
    cd = np.array([p == r['answer'] for p, r in zip(dense_pred, ev)], dtype=float)
    cr = np.array([p == r['answer'] for p, r in zip(routed, ev)], dtype=float)
    diff = cd - cr

    edges = np.linspace(0, 1, args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.9

    panels = [('all questions', np.ones(len(ev), bool)),
              ('OCR questions', is_ocr),
              ('shape-reasoning questions', ~is_ocr)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8),
                             gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    for col, (name, mask) in enumerate(panels):
        ax, axn = axes[0][col], axes[1][col]
        rows = binned_gap(alpha[mask], diff[mask], edges)
        vals = np.array([r[1] for r in rows])
        ses = np.array([r[2] for r in rows])
        ns = np.array([r[3] for r in rows])
        colors = ['#3b4a6b' if (v > 0) else '#c0392b' for v in np.nan_to_num(vals)]
        ax.bar(centers, np.nan_to_num(vals), width=width, color=colors,
               edgecolor='white', linewidth=0.5)
        ax.errorbar(centers, vals, yerr=ses, fmt='none', ecolor='0.35',
                    elinewidth=1, capsize=2)
        ax.axhline(0, color='0.2', linewidth=1)
        overall = 100 * diff[mask].mean()
        ax.axhline(overall, color='#e08a1e', linewidth=1.4, linestyle='--',
                   label=f'overall {overall:+.2f}')
        ax.set_title(name)
        ax.set_ylabel('dense - routed (points)')
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc='upper left')

        axn.bar(centers, ns, width=width, color='0.7')
        axn.set_yscale('log')
        axn.set_ylabel('samples')
        axn.set_xlabel(r'$\alpha$')
        axn.grid(axis='y', alpha=0.2, linewidth=0.5)
        axn.set_axisbelow(True)

    router = {'alpha': 'oracle alpha routing',
              'task': 'context / question-type routing',
              'mixture': 'random split, 0.5/0.5 token-wise mixture'}[args.route_by]
    fig.suptitle(f'dense minus routed accuracy vs '
                 r'$\alpha$'
                 f'  -- {router}'
                 f'  ({args.title}, {args.split} n={len(ev)}, '
                 f'{args.bins} bins of {1/args.bins:.2f})\n'
                 'positive = dense better;  error bars = +/-1 SE of the paired '
                 'per-sample difference', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'Saved {out}')

    print(f"\n{'alpha bin':<14}{'n':>6}{'dense':>8}{'routed':>8}{'gap':>8}{'SE':>7}")
    for b, v, se, n in binned_gap(alpha, diff, edges):
        if n == 0:
            continue
        m = np.clip(np.digitize(alpha, edges) - 1, 0, len(edges) - 2) == b
        print(f"[{edges[b]:.2f},{edges[b+1]:.2f})".ljust(14)
              + f"{n:>6}{100*cd[m].mean():>8.2f}{100*cr[m].mean():>8.2f}"
              + f"{v:>+8.2f}{se:>7.2f}")
    print(f"{'overall':<14}{len(ev):>6}{100*cd.mean():>8.2f}{100*cr.mean():>8.2f}"
          f"{100*diff.mean():>+8.2f}{100*diff.std(ddof=1)/np.sqrt(len(ev)):>7.2f}")


if __name__ == '__main__':
    main()
