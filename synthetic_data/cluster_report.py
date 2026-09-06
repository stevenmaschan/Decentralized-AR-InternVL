#!/usr/bin/env python3
"""
Characterise a k=2 balanced spherical k-means partition of the synthetic dataset.

Answers the question the partition exists to answer: *what* does it separate?
Compares the two clusters on every generative factor (alpha, shape count, text
presence/length/size/opacity, shape size/opacity), reports mutual information
between each factor and the cluster label, and checks how the OCR vs
shape-reasoning QA pairs would be split if experts were routed by this partition.

Usage:
    python synthetic_data/cluster_report.py \
        --clustering-dir synthetic_data/clip/balanced_kmeans_k2 \
        --features synthetic_data/clip/features_vit_base_patch16.npy \
        --metadata synthetic_data/metadata.jsonl \
        --annotations synthetic_data/annotations.jsonl
"""

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score, silhouette_score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clustering-dir', type=str,
                    default='synthetic_data/clip/balanced_kmeans_k2')
    ap.add_argument('--features', type=str,
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--keys', type=str, default=None)
    ap.add_argument('--metadata', type=str, default='synthetic_data/metadata.jsonl')
    ap.add_argument('--annotations', type=str,
                    default='synthetic_data/annotations.jsonl')
    ap.add_argument('--tsne-coords', type=str,
                    default='synthetic_data/clip/tsne_alpha_coords.npy',
                    help='reuse the t-SNE embedding from tsne_alpha.py (optional)')
    ap.add_argument('--output', type=str,
                    default='synthetic_data/clip/balanced_kmeans_k2/cluster_report.png')
    ap.add_argument('--silhouette-samples', type=int, default=5000)
    args = ap.parse_args()

    cdir = Path(args.clustering_dir)
    assign = np.load(cdir / 'clustering_assignments.npy')
    feat_path = Path(args.features)
    keys_path = Path(args.keys) if args.keys else \
        feat_path.with_name(feat_path.stem + '_keys.json')
    keys = json.load(open(keys_path))
    assert len(keys) == len(assign), (len(keys), len(assign))

    meta = {}
    for line in open(args.metadata):
        m = json.loads(line)
        meta[str(m['image_id'])] = m
    recs = [meta[k] for k in keys]

    n_clusters = int(assign.max()) + 1
    sizes = Counter(assign.tolist())
    print(f'{len(assign)} images, {n_clusters} clusters, sizes: '
          + ', '.join(f'c{c}={sizes[c]}' for c in range(n_clusters)))

    def per_image(fn):
        return np.array([fn(r) for r in recs], dtype=float)

    alpha = per_image(lambda r: r['alpha'])
    factors = [
        ('alpha', alpha),
        ('n_shapes', per_image(lambda r: r['n_shapes'])),
        ('has_text (frac)', per_image(lambda r: 1.0 if r['text'] else 0.0)),
        ('text length', per_image(lambda r: len(r['text']['string'])
                                  if r['text'] else np.nan)),
        ('text size px', per_image(lambda r: r['text']['size']
                                   if r['text'] else np.nan)),
        ('text opacity', per_image(lambda r: r['text']['opacity']
                                   if r['text'] else np.nan)),
        ('shape radius px', per_image(
            lambda r: np.mean([s['radius'] for s in r['shapes']]))),
        ('shape opacity', per_image(
            lambda r: np.mean([s['opacity'] for s in r['shapes']]))),
    ]

    head = 'factor (mean)'.ljust(20) + ''.join(f'c{c}'.rjust(11)
                                               for c in range(n_clusters))
    print('\n' + head)
    print('-' * len(head))
    for name, v in factors:
        row = ''.join(f'{np.nanmean(v[assign == c]):>11.3f}'
                      for c in range(n_clusters))
        print(name.ljust(20) + row)

    # Mutual information: how much does knowing a factor tell you the cluster?
    print('\nmutual information with cluster label '
          f'(bits, max {np.log2(n_clusters):.1f})')
    mi_factors = [
        ('alpha (5 bins)', np.digitize(alpha, [.2, .4, .6, .8])),
        ('n_shapes', per_image(lambda r: r['n_shapes']).astype(int)),
        ('has_text', per_image(lambda r: 1 if r['text'] else 0).astype(int)),
        ('text length', per_image(lambda r: len(r['text']['string'])
                                  if r['text'] else 0).astype(int)),
    ]
    mi = {}
    for name, lab in mi_factors:
        mi[name] = mutual_info_score(lab, assign) / np.log(2)
        print(f'  {name:<18}{mi[name]:.3f}')

    # Cluster quality in the space k-means actually used (mean-subtracted, L2).
    mean_file = cdir / 'clustering_global_mean.npy'
    X = np.load(feat_path)
    if mean_file.exists():
        X = X - np.load(mean_file)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    n_sil = min(args.silhouette_samples, len(X))
    idx = np.random.RandomState(0).choice(len(X), n_sil, replace=False)
    sil = silhouette_score(X[idx], assign[idx], metric='cosine')
    print(f'\nsilhouette (cosine, {n_sil} samples): {sil:.3f}')

    # How the QA pairs would split if an expert were picked per image.
    cluster_of = {int(k): int(assign[i]) for i, k in enumerate(keys)}
    tally = Counter()
    for line in open(args.annotations):
        a = json.loads(line)
        tally[(cluster_of[a['image_id']], a['task_type'])] += 1
    print('\nQA pairs per cluster if experts were routed by this partition')
    for c in range(n_clusters):
        o = tally[(c, 'ocr')]
        sr = tally[(c, 'shape_reasoning')]
        print(f'  c{c}: ocr={o} shape_reasoning={sr} '
              f'({100 * o / max(1, o + sr):.1f}% OCR)')

    # --- figure -------------------------------------------------------------
    coords = None
    tp = Path(args.tsne_coords)
    if tp.exists():
        c = np.load(tp)
        if len(c) == len(assign):
            coords = c
    n_panels = 3 if coords is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.4))
    palette = ['#2a6fdb', '#e5893b', '#3fa45b', '#d94a3d']

    ax = axes[0]
    for c in range(n_clusters):
        ax.hist(alpha[assign == c], bins=25, range=(0, 1), alpha=0.6,
                color=palette[c % len(palette)], label=f'cluster {c} '
                f'(mean {alpha[assign == c].mean():.2f})')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('images')
    ax.set_title(r'$\alpha$ distribution per cluster'
                 f"  (MI {mi['alpha (5 bins)']:.3f} bits)")
    ax.legend(fontsize=9)

    ax = axes[1]
    width = 0.8 / n_clusters
    counts = np.arange(1, 11)
    for c in range(n_clusters):
        d = Counter(per_image(lambda r: r['n_shapes'])[assign == c].astype(int).tolist())
        ax.bar(counts + (c - (n_clusters - 1) / 2) * width,
               [d[i] for i in counts], width=width,
               color=palette[c % len(palette)], label=f'cluster {c}')
    ax.set_xlabel('number of shapes')
    ax.set_ylabel('images')
    ax.set_xticks(counts)
    ax.set_title(f"shape count per cluster  (MI {mi['n_shapes']:.3f} bits)")
    ax.legend(fontsize=9)

    if coords is not None:
        ax = axes[2]
        for c in range(n_clusters):
            m = assign == c
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.55,
                       color=palette[c % len(palette)], linewidths=0,
                       label=f'cluster {c}')
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.set_title('clusters on the t-SNE embedding')
        ax.legend(fontsize=9, markerscale=2)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
