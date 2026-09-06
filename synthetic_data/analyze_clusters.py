#!/usr/bin/env python3
"""
Describe what a k-means partition of the CLIP features actually separates.

Loads the assignments written by clustering/single_stage_balanced_kmeans.py,
joins them to the dataset ground truth, and reports per-cluster means of every
generation factor plus how well each factor predicts the partition. Reuses the
t-SNE coordinates cached by tsne_alpha.py so the panels are directly comparable
to tsne_alpha.png.

Usage:
    python synthetic_data/analyze_clusters.py \
        --assignments synthetic_data/clip/balanced-kmeans_2/clustering_assignments.npy \
        --output synthetic_data/clip/balanced-kmeans_2/clusters.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, adjusted_rand_score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--assignments', type=str,
                    default='synthetic_data/clip/balanced-kmeans_2/clustering_assignments.npy')
    ap.add_argument('--keys', type=str,
                    default='synthetic_data/clip/features_vit_base_patch16_keys.json')
    ap.add_argument('--metadata', type=str, default='synthetic_data/metadata.jsonl')
    ap.add_argument('--coords', type=str,
                    default='synthetic_data/clip/tsne_alpha_coords.npy',
                    help='t-SNE coordinates cached by tsne_alpha.py')
    ap.add_argument('--output', type=str,
                    default='synthetic_data/clip/balanced-kmeans_2/clusters.png')
    ap.add_argument('--dpi', type=int, default=160)
    args = ap.parse_args()

    assign = np.load(args.assignments)
    keys = json.load(open(args.keys))
    assert len(keys) == len(assign), (len(keys), len(assign))

    meta = {}
    for line in open(args.metadata):
        m = json.loads(line)
        meta[str(m['image_id'])] = m
    recs = [meta[k] for k in keys]

    nan = float('nan')
    factors = {
        'alpha': np.array([r['alpha'] for r in recs]),
        'shape count': np.array([r['n_shapes'] for r in recs], dtype=float),
        'mean shape radius': np.array([np.mean([s['radius'] for s in r['shapes']])
                                       for r in recs]),
        'mean shape opacity': np.array([np.mean([s['opacity'] for s in r['shapes']])
                                        for r in recs]),
        'text size': np.array([r['text']['size'] if r['text'] else nan for r in recs]),
        'text opacity': np.array([r['text']['opacity'] if r['text'] else nan
                                  for r in recs]),
        'text length': np.array([len(r['text']['string']) if r['text'] else nan
                                 for r in recs]),
    }
    has_text = np.array([r['text'] is not None for r in recs])
    labels = sorted(set(assign.tolist()))

    print(f'{len(assign)} samples, {len(labels)} clusters, '
          f'sizes {[int((assign == c).sum()) for c in labels]}')
    head = 'factor'.ljust(22) + ''.join(f'cluster {c}'.rjust(12) for c in labels)
    print('\n' + head)
    print('-' * len(head))
    for name, v in factors.items():
        row = name.ljust(22)
        for c in labels:
            x = v[assign == c]
            x = x[~np.isnan(x)]
            row += f'{x.mean():>12.3f}' if len(x) else f'{"n/a":>12}'
        print(row)
    row = 'images with text %'.ljust(22)
    for c in labels:
        row += f'{100 * has_text[assign == c].mean():>12.1f}'
    print(row)

    # Which factor separates the partition? (binary partitions only)
    if len(labels) == 2:
        print('\nhow well each factor predicts the partition '
              '(AUC 0.5 = no signal, 1.0 = perfect):')
        rows = []
        for name, v in factors.items():
            m = ~np.isnan(v)
            auc = roc_auc_score(assign[m], v[m])
            rows.append((max(auc, 1 - auc), name, auc, int(m.sum())))
        for _, name, auc, n in sorted(rows, reverse=True):
            print(f'  {name:<22} AUC={auc:.3f}   (n={n})')
        ari = adjusted_rand_score(has_text, assign)
        print(f'  {"presence of text":<22} ARI={ari:+.3f}   '
              f'({int((~has_text & (assign == labels[0])).sum())} of '
              f'{int((~has_text).sum())} text-free images in cluster {labels[0]})')

    # --- panels -------------------------------------------------------------
    coords = np.load(args.coords) if Path(args.coords).exists() else None
    n_panels = 3 if coords is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 5.6))
    axes = np.atleast_1d(axes)
    palette = ['#2a6fdb', '#d94a3d', '#2e9e5b', '#b07cd6', '#e0912a']

    if coords is not None:
        for i, c in enumerate(labels):
            m = assign == c
            axes[0].scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.6,
                            c=palette[i % len(palette)],
                            label=f'cluster {c} (n={int(m.sum())})', linewidths=0)
        axes[0].legend(loc='best', fontsize=9, markerscale=2)
        axes[0].set_title('t-SNE colored by balanced k-means cluster')

        sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=factors['alpha'],
                             cmap='viridis', s=8, alpha=0.7, linewidths=0)
        fig.colorbar(sc, ax=axes[1]).set_label(r'$\alpha$')
        axes[1].set_title(r'the same embedding colored by $\alpha$')
        for a in axes[:2]:
            a.set_xlabel('t-SNE dim 1')
            a.set_ylabel('t-SNE dim 2')
            a.grid(alpha=0.15, linewidth=0.5)
        ax_hist = axes[2]
    else:
        ax_hist = axes[0]

    bins = np.linspace(0, 1, 26)
    for i, c in enumerate(labels):
        ax_hist.hist(factors['alpha'][assign == c], bins=bins, alpha=0.6,
                     color=palette[i % len(palette)], label=f'cluster {c}')
    ax_hist.set_xlabel(r'$\alpha$')
    ax_hist.set_ylabel('images')
    ax_hist.legend(fontsize=9)
    ax_hist.set_title(r'$\alpha$ distribution per cluster')
    ax_hist.grid(alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
