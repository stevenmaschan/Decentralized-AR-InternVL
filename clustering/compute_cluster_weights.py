#!/usr/bin/env python3
"""
Compute soft cluster weights for each training sample from the trained clustering.

For every image feature f and each cluster centroid c_k:
    s_k = cosine_similarity(f, c_k)            # f, c_k L2-normalised -> dot product
    w_k = softmax_k(s)  = exp(s_k) / sum_j exp(s_j)

w is a probability distribution over clusters (the "cluster weight" of the sample).
Weights are computed once per unique image and every training line (dense JSONL row)
inherits its image's weights via image_key().

Outputs (in --output-dir):
    <prefix>_cluster_weights.npy   float32 (N_unique, K) aligned to the features' keys
    cluster_weight_statistics.txt  human-readable statistics
    cluster_weight_confidence.png  histogram of per-sample confidence (max weight)

Usage:
    python clustering/compute_cluster_weights.py \
        --features-prefix clustering/unique_features_vit_base_patch16 \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42 \
        --dense-dir /lambda/nfs/virginia/internvl-data/dense \
        --output-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_unique_image_jsonl import image_key


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-prefix', default='clustering/unique_features_vit_base_patch16')
    ap.add_argument('--clustering-dir',
                    default='clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42')
    ap.add_argument('--dense-dir', default='/lambda/nfs/virginia/internvl-data/dense')
    ap.add_argument('--output-dir',
                    default='clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42')
    ap.add_argument('--prefix', default='clustering')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- load features, keys, centroids, hard assignments ----
    feats = np.load(args.features_prefix + '.npy')
    keys = json.load(open(args.features_prefix + '_keys.json'))
    centroids = np.load(Path(args.clustering_dir) / f'{args.prefix}_centroids.npy')
    hard = np.load(Path(args.clustering_dir) / f'{args.prefix}_assignments.npy')
    K = centroids.shape[0]
    N = feats.shape[0]
    print(f"features {feats.shape}, {N} keys, {K} centroids")

    # ---- subtract the same global mean used during clustering (if any) ----
    # The centroids live in the mean-subtracted, L2-normalized space, so features
    # must be centered identically before measuring cosine similarity.
    mean_file = Path(args.clustering_dir) / f'{args.prefix}_global_mean.npy'
    if mean_file.exists():
        mean_vector = np.load(mean_file).astype(np.float64)
        feats = feats.astype(np.float64) - mean_vector
        print(f"Subtracted global mean (||mean|| = {np.linalg.norm(mean_vector):.4f}) before cosine sims")
    else:
        feats = feats.astype(np.float64)
        print("No global mean found; computing cosine sims on raw (non-centered) features")

    # ---- cosine sims + softmax weights ----
    sims = l2norm(feats) @ l2norm(centroids.astype(np.float64)).T   # (N, K)
    weights = softmax(sims, axis=1).astype(np.float32)                                  # (N, K)

    np.save(out / f'{args.prefix}_cluster_weights.npy', weights)
    key_to_w = {k: weights[i] for i, k in enumerate(keys)}

    # ---- overall statistics ----
    soft_arg = weights.argmax(1)
    conf = weights.max(1)                                    # confidence = max weight
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1)
    max_entropy = np.log(K)
    agree_hard = float((soft_arg == hard).mean())

    lines = []
    def P(s=""): lines.append(s); print(s)

    P("=" * 90)
    P("CLUSTER WEIGHT STATISTICS  (w = softmax over clusters of cosine similarity to centroids)")
    P("=" * 90)
    P(f"Unique images (samples with features) : {N:,}")
    P(f"Clusters (K)                          : {K}")
    P(f"Centroid-centroid cosine similarity   : {float(l2norm(centroids)[0] @ l2norm(centroids)[1]):.4f}")
    P("")
    P("Cosine similarity to centroids (raw, pre-softmax):")
    for k in range(K):
        P(f"  s(centroid {k}): min={sims[:,k].min():.4f}  mean={sims[:,k].mean():.4f}  max={sims[:,k].max():.4f}")
    P(f"  |s0 - s1|      : mean={np.abs(sims[:,0]-sims[:,1]).mean():.4f}  max={np.abs(sims[:,0]-sims[:,1]).max():.4f}")
    P("")
    P("Confidence = max cluster weight per sample:")
    for q in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        P(f"  p{q:<3d} = {np.percentile(conf, q):.4f}")
    P(f"  mean = {conf.mean():.4f}   (0.5 = fully ambiguous, 1.0 = certain)")
    P("")
    P("Fraction of samples above confidence thresholds:")
    for t in [0.51, 0.55, 0.60, 0.70, 0.80, 0.90]:
        P(f"  conf > {t:.2f} : {float((conf > t).mean())*100:6.2f}%")
    P("")
    P(f"Mean normalised entropy H/Hmax : {entropy.mean()/max_entropy:.4f}  (1.0 = uniform weights)")
    P("")
    P("Soft (argmax of weights) vs hard balanced-kmeans assignment:")
    P(f"  agreement           : {agree_hard*100:.2f}%")
    P(f"  soft cluster sizes  : {np.bincount(soft_arg, minlength=K).tolist()}")
    P(f"  hard cluster sizes  : {np.bincount(hard, minlength=K).tolist()}")
    P("")

    # ---- per-dataset statistics at the TRAINING-LINE level ----
    P("=" * 90)
    P("PER-DATASET (training-line level; each line weighted by its image's cluster weights)")
    P("=" * 90)
    header = f"{'dataset':30s} {'lines':>9} {'mean w0':>9} {'mean w1':>9} {'mean conf':>10} {'soft->c0':>9} {'soft->c1':>9}"
    P(header); P("-" * len(header))
    dense = Path(args.dense_dir)
    tot = np.zeros(K); tot_conf = 0.0; tot_lines = 0; tot_soft = np.zeros(K, dtype=int)
    for jf in sorted(dense.glob('*.jsonl')):
        sw = np.zeros(K); sc = 0.0; n = 0; ssoft = np.zeros(K, dtype=int)
        with open(jf) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try: img = json.loads(line).get('image')
                except json.JSONDecodeError: continue
                if not img: continue
                w = key_to_w.get(image_key(img.strip().replace('\\', '/')))
                if w is None: continue
                sw += w; sc += w.max(); ssoft[int(w.argmax())] += 1; n += 1
        if n == 0: continue
        tot += sw; tot_conf += sc; tot_lines += n; tot_soft += ssoft
        P(f"{jf.stem:30s} {n:9d} {sw[0]/n:9.4f} {sw[1]/n:9.4f} {sc/n:10.4f} {ssoft[0]:9d} {ssoft[1]:9d}")
    P("-" * len(header))
    P(f"{'TOTAL':30s} {tot_lines:9d} {tot[0]/tot_lines:9.4f} {tot[1]/tot_lines:9.4f} "
      f"{tot_conf/tot_lines:10.4f} {tot_soft[0]:9d} {tot_soft[1]:9d}")

    (out / 'cluster_weight_statistics.txt').write_text("\n".join(lines) + "\n")
    print(f"\nSaved stats -> {out/'cluster_weight_statistics.txt'}")
    print(f"Saved weights -> {out/(args.prefix + '_cluster_weights.npy')}")

    # ---- confidence histogram ----
    plt.figure(figsize=(9, 5))
    plt.hist(conf, bins=100, color='#3b7dd8', edgecolor='none')
    plt.axvline(0.5, color='k', ls='--', lw=1, label='0.5 (ambiguous)')
    plt.xlabel('Confidence = max cluster weight'); plt.ylabel('# unique images')
    plt.title('Soft cluster-weight confidence (softmax of cosine sim to centroids)')
    plt.legend(); plt.tight_layout()
    plt.savefig(out / 'cluster_weight_confidence.png', dpi=150)
    print(f"Saved plot -> {out/'cluster_weight_confidence.png'}")


if __name__ == '__main__':
    main()
