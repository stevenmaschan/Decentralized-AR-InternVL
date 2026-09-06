#!/usr/bin/env python3
"""
t-SNE of the synthetic dataset's CLIP features, colored by the alpha parameter.

Takes the CLIP ViT-B/16 features produced by clustering/extract_clip_features.py
(rows aligned with <features>_keys.json, whose keys are the metadata image_ids),
embeds them in 2-D with cosine t-SNE, and colors every point by the image's
``alpha`` -- the single scalar that trades text prominence against shape
prominence at generation time.

It also reports how strongly alpha is encoded in the CLIP space, and -- because
CLIP is extremely sensitive to rendered text -- how much of the space is instead
explained by the identity of the word written on the image:
  * Spearman correlation between alpha and each t-SNE axis
  * k-NN regression R^2 of alpha from the raw 512-d features (5-fold CV)
  * 1-NN string purity and the share of feature variance explained by
    string identity / alpha / shape count / text presence / text length
    (the two string-identity statistics are reported only when strings repeat;
    with unique random strings they are degenerate)

Usage:
    python synthetic_data/tsne_alpha.py \
        --features synthetic_data/clip/features_vit_base_patch16.npy \
        --metadata synthetic_data/metadata.jsonl \
        --output synthetic_data/clip/tsne_alpha.png
"""

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr


NO_TEXT = '<no text>'


def normalize_features(features):
    """Normalize features to unit length for cosine distance."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return features / norms


def variance_explained(X, labels):
    """Fraction of total feature variance explained by a categorical factor."""
    total = ((X - X.mean(axis=0)) ** 2).sum()
    within = 0.0
    for v in np.unique(labels):
        g = X[labels == v]
        within += ((g - g.mean(axis=0)) ** 2).sum()
    return 1.0 - within / total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', type=str,
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--keys', type=str, default=None,
                    help='aligned *_keys.json (default: <features stem>_keys.json)')
    ap.add_argument('--metadata', type=str, default='synthetic_data/metadata.jsonl')
    ap.add_argument('--output', type=str, default='synthetic_data/clip/tsne_alpha.png')
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--max-iter', type=int, default=1000)
    ap.add_argument('--cmap', type=str, default='viridis')
    ap.add_argument('--dpi', type=int, default=160)
    args = ap.parse_args()

    feat_path = Path(args.features)
    keys_path = Path(args.keys) if args.keys else \
        feat_path.with_name(feat_path.stem + '_keys.json')

    features = np.load(feat_path)
    keys = json.load(open(keys_path))
    assert len(keys) == features.shape[0], (len(keys), features.shape)
    print(f'Loaded features {features.shape} from {feat_path}')

    meta = {}
    for line in open(args.metadata):
        m = json.loads(line)
        meta[str(m['image_id'])] = m
    missing = [k for k in keys if k not in meta]
    assert not missing, f'{len(missing)} feature rows have no metadata (e.g. {missing[:3]})'

    alpha = np.array([meta[k]['alpha'] for k in keys], dtype=np.float32)
    n_shapes = np.array([meta[k]['n_shapes'] for k in keys])
    has_text = np.array([meta[k]['text'] is not None for k in keys])
    word = np.array([meta[k]['text']['string'].lower() if meta[k]['text'] else NO_TEXT
                     for k in keys])
    print(f'alpha: min={alpha.min():.3f} median={np.median(alpha):.3f} max={alpha.max():.3f}')

    # Cosine t-SNE, matching clustering/generate_tsne_plot.py conventions.
    normed = normalize_features(features)
    print(f'Running t-SNE (perplexity={args.perplexity}, cosine)...')
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(normed) - 1),
        max_iter=args.max_iter,
        metric='cosine',
        init='pca',
        random_state=args.random_state,
        verbose=1,
    )
    coords = tsne.fit_transform(normed)

    # --- how much of alpha does the CLIP space actually carry? ---------------
    r1 = spearmanr(alpha, coords[:, 0]).statistic
    r2 = spearmanr(alpha, coords[:, 1]).statistic
    knn_r2 = cross_val_score(
        KNeighborsRegressor(n_neighbors=10, metric='cosine'),
        normed, alpha, cv=5, scoring='r2').mean()
    print(f'\nSpearman(alpha, t-SNE dim 1) = {r1:+.3f}')
    print(f'Spearman(alpha, t-SNE dim 2) = {r2:+.3f}')
    print(f'k-NN (k=10, cosine) 5-fold CV R^2 predicting alpha from CLIP features = {knn_r2:.3f}')

    # What the neighbourhoods are actually made of. `variance_explained` is only
    # meaningful for a factor with repeated levels: when every rendered string is
    # unique each group has one member, within-group variance is 0 by
    # construction, and the statistic degenerates to ~100%. Same for 1-NN string
    # purity, which is then ~0 by construction (ignoring the no-text images).
    text_only = has_text
    n_unique = len(set(word[text_only]))
    n_text = int(text_only.sum())
    # A handful of chance collisions among random strings is not enough: the
    # statistic is only informative once the average group actually has several
    # members, otherwise nearly every group is a singleton with zero within-group
    # variance and the number is ~100% by construction.
    strings_repeat = n_unique <= n_text / 2

    sim = normed @ normed.T
    np.fill_diagonal(sim, -2.0)
    nn1 = sim.argmax(axis=1)

    print(f'\nrendered strings: {n_text} total, {n_unique} unique')
    purity = float((word[nn1] == word)[text_only].mean())
    print(f'1-NN shares the same string (text images) : {100 * purity:.1f}%')
    if not strings_repeat:
        print(f'  (strings are effectively unique -- {n_unique} distinct over '
              f'{n_text} images -- so string-identity variance-explained is')
        print('   degenerate and is not reported)')

    factors = [('alpha (5 equal bins)', np.digitize(alpha, [.2, .4, .6, .8])),
               ('shape count', n_shapes),
               ('presence of text', has_text),
               ('text length', np.array([len(w) if w != NO_TEXT else 0
                                         for w in word]))]
    if strings_repeat:
        factors.insert(0, ('string identity', word))
    print('share of feature variance explained by')
    for name, labels in factors:
        print(f'  {name:<22}: {100 * variance_explained(normed, labels):5.1f}%')

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=alpha, cmap=args.cmap,
                    s=26, alpha=0.9, linewidths=0.3, edgecolors='white')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\alpha$   (0 = large opaque shapes / small faint text'
                   '  $\\rightarrow$  1 = small faint shapes / large opaque text)',
                   fontsize=10)
    ax.set_title(f'CLIP ViT-B/16 t-SNE of {len(coords)} synthetic images, '
                 r'colored by $\alpha$', fontsize=13)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.grid(alpha=0.15, linewidth=0.5)
    caption = (fr'Spearman($\alpha$, dim1)={r1:+.2f}    '
               fr'Spearman($\alpha$, dim2)={r2:+.2f}    '
               fr'k-NN $R^2$($\alpha$)={knn_r2:.2f}')
    if purity is not None:
        caption += fr'    1-NN string purity={100 * purity:.0f}%'
    ax.text(0.01, 0.01, caption, transform=ax.transAxes, fontsize=9, color='0.35')
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'\nSaved {out}')

    # Companion panels: the same embedding colored by the factors that compete
    # with alpha, so the alpha gradient can be read in context.
    panels = []
    if strings_repeat:
        panels.append('string')
    panels += ['n_shapes', 'has_text', 'text_len']
    fig2, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.8))
    axes = np.atleast_1d(axes)

    for ax2, panel in zip(axes, panels):
        if panel == 'string':
            top = [w for w, _ in Counter(word[text_only]).most_common(10)]
            ax2.scatter(coords[:, 0], coords[:, 1], s=18, c='0.85', linewidths=0)
            cmap10 = plt.get_cmap('tab10')
            for i, w in enumerate(top):
                m = word == w
                ax2.scatter(coords[m, 0], coords[m, 1], s=34, color=cmap10(i),
                            label=f'{w} (n={int(m.sum())})', linewidths=0.3,
                            edgecolors='white')
            ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8,
                       frameon=False)
            ax2.set_title(f'colored by string identity (10 most frequent)\n'
                          f'1-NN string purity = {100 * purity:.0f}%')
        elif panel == 'n_shapes':
            h = ax2.scatter(coords[:, 0], coords[:, 1], c=n_shapes, cmap='plasma',
                            s=22, alpha=0.9, linewidths=0.2, edgecolors='white')
            fig2.colorbar(h, ax=ax2).set_label('number of shapes')
            ax2.set_title('colored by shape count')
        elif panel == 'has_text':
            for lbl, mask, color in (('with text', has_text, '#2a6fdb'),
                                     ('no text', ~has_text, '#d94a3d')):
                ax2.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.85,
                            c=color, label=f'{lbl} (n={int(mask.sum())})',
                            linewidths=0.2, edgecolors='white')
            ax2.legend(loc='best', fontsize=9)
            ax2.set_title('colored by presence of text')
        else:
            tl = np.array([len(w) if w != NO_TEXT else 0 for w in word])
            m = tl > 0
            ax2.scatter(coords[~m, 0], coords[~m, 1], s=18, c='0.85',
                        linewidths=0, label='no text')
            h = ax2.scatter(coords[m, 0], coords[m, 1], c=tl[m], cmap='cividis',
                            s=22, alpha=0.9, linewidths=0.2, edgecolors='white')
            fig2.colorbar(h, ax=ax2).set_label('letters in the string')
            ax2.set_title('colored by text length')

    for a in axes:
        a.set_xlabel('t-SNE dim 1')
        a.set_ylabel('t-SNE dim 2')
        a.grid(alpha=0.15, linewidth=0.5)
    fig2.tight_layout()
    out2 = out.with_name(out.stem + '_factors' + out.suffix)
    fig2.savefig(out2, dpi=args.dpi)
    print(f'Saved {out2}')

    np.save(out.with_name(out.stem + '_coords.npy'), coords)
    print(f'Saved {out.with_name(out.stem + "_coords.npy")}')


if __name__ == '__main__':
    main()
