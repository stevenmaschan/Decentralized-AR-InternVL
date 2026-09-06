#!/usr/bin/env python3
"""Plot dense vs routed accuracy against the generative parameter alpha.

Reads a routed_results json (written by eval_routed.py) and draws exact match per
alpha bin for the two systems being compared -- the single dense model and the
oracle-alpha-routed pair -- overall and split into the two task families. The
number above each bin is the signed routed-minus-dense gap.

    python -m synthetic_data.vlm.plot_alpha_bins \
        --results synthetic_data/vlm/runs/seed0_archive/routed_val.json \
        --output synthetic_data/vlm/runs/seed0_archive/alpha_bins.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {                       # label, colour
    'dense':     ('dense', '#3b4a6b'),
    'routed':    ('routed (oracle alpha)', '#c0392b'),
    'expert_lo': ('expert_lo (alpha<0.5)', '#2a8fbd'),
    'expert_hi': ('expert_hi (alpha>=0.5)', '#e07b39'),
}


def series(rep, key):
    """(bin index, value, n) for a metric, skipping bins the model does not cover."""
    out = []
    for b, e in sorted(rep['per_alpha_bin'].items(), key=lambda kv: int(kv[0])):
        v = e['exact_match'] if key == 'exact_match' else e.get(key)
        n = e['n'] if key == 'exact_match' else e.get(key.replace('_em', '_n'), 0)
        if v is not None and n:
            out.append((int(b), 100 * v, n))
    return out


def panel(ax, res, key, title, models, annotate_n=True):
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        if m not in res:
            continue
        pts = series(res[m], key)
        if not pts:
            continue
        xs = np.array([p[0] for p in pts]) + (i - (len(models) - 1) / 2) * width
        ys = [p[1] for p in pts]
        label, colour = STYLE[m]
        bars = ax.bar(xs, ys, width=width, color=colour, label=label,
                      edgecolor='white', linewidth=0.6)
        if annotate_n:
            for bar, p in zip(bars, pts):
                ax.text(bar.get_x() + bar.get_width() / 2, p[1] + 0.8, str(p[2]),
                        ha='center', va='bottom', fontsize=6, color='0.35',
                        rotation=90)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'{b/5:.1f}-{(b+1)/5:.1f}' for b in range(5)])
    ax.set_xlabel(r'$\alpha$ bin')
    ax.set_ylabel('exact match (%)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results',
                    default='synthetic_data/vlm/runs/seed0_archive/routed_val.json')
    ap.add_argument('--output',
                    default='synthetic_data/vlm/runs/seed0_archive/alpha_bins.png')
    ap.add_argument('--split-name', default='val')
    args = ap.parse_args()

    res = json.load(open(args.results))
    models = ['dense', 'routed']
    # eval_routed records which router produced these numbers; label accordingly.
    if res.get('route_by') == 'task':
        STYLE['routed'] = ('routed (context / question-type)', '#1a7a3c')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    panel(axes[0], res, 'exact_match', 'all questions', models)
    panel(axes[1], res, 'ocr_em', 'OCR questions only', models)
    panel(axes[2], res, 'shape_em', 'shape-reasoning questions only', models)
    # One shared legend below the panels, so the per-bin gap labels along the top
    # of each axes have the whole width to themselves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    # Signed gap per bin, so the direction of the difference is readable.
    for ax, key in zip(axes, ('exact_match', 'ocr_em', 'shape_em')):
        d = {b: v for b, v, _ in series(res['dense'], key)}
        r = {b: v for b, v, _ in series(res['routed'], key)}
        top = max(list(d.values()) + list(r.values()))
        ax.set_ylim(0, top * 1.22)
        for b in sorted(set(d) & set(r)):
            gap = r[b] - d[b]
            ax.text(b, top * 1.13, f'{gap:+.1f}', ha='center', fontsize=9,
                    color=('#1a7a3c' if gap > 0 else '#b3271e'), weight='bold')

    n = res['dense']['n']
    router = {'task': 'context (question-type) routing'}.get(
        res.get('route_by'), 'oracle alpha routing')
    fig.suptitle(f'{router}:  accuracy vs the generative parameter '
                 r'$\alpha$'
                 f'  ({args.split_name} split, n={n})\n'
                 'bold numbers = routed minus dense (points);  '
                 'small numbers = sample count',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'Saved {out}')

    print(f"\n{'bin':<10}{'n':>6}{'dense':>9}{'routed':>9}{'gap':>8}")
    for b in range(5):
        e = res['dense']['per_alpha_bin'].get(str(b))
        if not e:
            continue
        d = 100 * e['exact_match']
        r = 100 * res['routed']['per_alpha_bin'][str(b)]['exact_match']
        print(f"{f'{b/5:.1f}-{(b+1)/5:.1f}':<10}{e['n']:>6}{d:>9.1f}{r:>9.1f}"
              f"{r - d:>+8.1f}")
    dn = res['dense']['n']
    do, ro = 100 * res['dense']['exact_match'], 100 * res['routed']['exact_match']
    print(f"{'overall':<10}{dn:>6}{do:>9.2f}{ro:>9.2f}{ro - do:>+8.2f}")


if __name__ == '__main__':
    main()
