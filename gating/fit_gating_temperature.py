#!/usr/bin/env python3
"""
Fit the gating temperature T that maximizes the dataset log-likelihood of a
mixture of cluster experts, using the precomputed per-sequence expert
log-likelihoods (see precompute_expert_loglik.py).

Model (temperature is MULTIPLIED with the gating logits):

    w_k(x; T) = softmax_k( T * s_k(x) )                 # gating over experts
    p(x; T)   = sum_k w_k(x; T) * p_k(x)                # mixture likelihood
    L(T)      = sum_x r_x * log p(x; T)                  # weighted dataset LL

where s_k(x) = cosine similarity of x's image to centroid k (gating_logits),
p_k(x) = exp(loglik[x, k]) is the sequence likelihood under expert k, and r_x is
the sequence weight (repeat_time). In log space, per sequence:

    log p(x; T) = logsumexp_k( log_softmax_k(T * s)_k + loglik[x, k] )

T -> 0 gives uniform gating; T -> inf gives hard argmax gating. We maximize L(T)
over T >= 0 (a smooth 1-D problem).

Requires the expert index k to align with the centroid/cluster index k
(expert k is the expert for cluster k). This is the case when the precompute is
run with --expert-checkpoints listed in cluster order.

Usage:
    /lambda/nfs/virginia/clip-feat-venv/bin/python gating/fit_gating_temperature.py \
        --loglik gating/expert_loglik_debug.npz \
        --output-dir gating/temperature_fit_debug
"""

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, log_softmax
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dataset_loglik(T, gating_logits, loglik, weights):
    """Weighted dataset log-likelihood at temperature T (scalar)."""
    logw = log_softmax(T * gating_logits, axis=1)          # (N, K)
    logp = logsumexp(logw + loglik, axis=1)                # (N,)
    return float(np.sum(weights * logp))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--loglik', required=True,
                    help='.npz from precompute_expert_loglik.py')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--t-max', type=float, default=100.0,
                    help='upper bound on T for the search (default: 100)')
    ap.add_argument('--no-weight', action='store_true',
                    help='ignore repeat_time and weight every sequence equally')
    ap.add_argument('--length-normalize', action='store_true',
                    help='divide each sequence loglik by its #response tokens '
                         '(per-token geometric-mean likelihood; changes the model)')
    args = ap.parse_args()

    d = np.load(args.loglik)
    if 'gating_logits' not in d:
        raise SystemExit('npz has no gating_logits — re-run precompute with --clustering-dir')
    loglik = d['loglik'].astype(np.float64)                # (N, K_exp)
    gating_logits = d['gating_logits'].astype(np.float64)  # (N, K_clu)
    has_gating = d['has_gating'] if 'has_gating' in d else np.ones(len(loglik), bool)
    weights = np.ones(len(loglik)) if args.no_weight else d['repeat_time'].astype(np.float64)

    if loglik.shape[1] != gating_logits.shape[1]:
        raise SystemExit(
            f'expert count {loglik.shape[1]} != cluster count {gating_logits.shape[1]}; '
            'experts must align with centroids (list --expert-checkpoints in cluster order)')

    # keep rows with gating logits and finite expert loglik
    keep = has_gating & np.isfinite(loglik).all(1) & np.isfinite(gating_logits).all(1)
    dropped = int((~keep).sum())
    loglik, gating_logits, weights = loglik[keep], gating_logits[keep], weights[keep]
    N, K = loglik.shape
    if args.length_normalize:
        ntok = d['n_resp_tokens'].astype(np.float64)[keep].clip(1)
        loglik = loglik / ntok[:, None]
    print(f'Fitting on N={N} sequences, K={K} experts/clusters '
          f'({dropped} dropped for missing gating / non-finite loglik)')

    # ---- 1-D maximization of L(T) over T in [0, t_max] ----
    res = minimize_scalar(lambda T: -dataset_loglik(T, gating_logits, loglik, weights),
                          bounds=(0.0, args.t_max), method='bounded',
                          options={'xatol': 1e-4})
    T_star = float(res.x)
    L_star = dataset_loglik(T_star, gating_logits, loglik, weights)

    # reference points
    L_uniform = dataset_loglik(0.0, gating_logits, loglik, weights)   # T=0: equal gating
    L_T1 = dataset_loglik(1.0, gating_logits, loglik, weights)        # raw cosine sims
    L_hard = dataset_loglik(args.t_max, gating_logits, loglik, weights)  # ~argmax gating
    # best single expert (T -> assign everything to one expert)
    L_single = [float(np.sum(weights * loglik[:, k])) for k in range(K)]

    Wtot = float(weights.sum())
    def per_seq(x):
        return x / Wtot

    print('\n==================== TEMPERATURE FIT ====================')
    print(f'  T*                      = {T_star:.4f}')
    print(f'  L(T*)                   = {L_star:.2f}   (per-seq {per_seq(L_star):.4f})')
    print(f'  L(T=0,  uniform gating) = {L_uniform:.2f}   (per-seq {per_seq(L_uniform):.4f})')
    print(f'  L(T=1,  raw cosine)     = {L_T1:.2f}   (per-seq {per_seq(L_T1):.4f})')
    print(f'  L(T={args.t_max:g}, ~hard argmax) = {L_hard:.2f}   (per-seq {per_seq(L_hard):.4f})')
    for k in range(K):
        print(f'  single-expert {k}         = {L_single[k]:.2f}   (per-seq {per_seq(L_single[k]):.4f})')
    print(f'  gain over uniform       = {L_star - L_uniform:.2f}')
    print('========================================================')

    # ---- L(T) curve ----
    Ts = np.concatenate([[0.0], np.geomspace(1e-2, args.t_max, 120)])
    Ls = np.array([dataset_loglik(T, gating_logits, loglik, weights) for T in Ts])

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Ls, color='#3b7dd8')
    plt.axvline(T_star, color='k', ls='--', lw=1, label=f'T* = {T_star:.3f}')
    plt.axhline(L_uniform, color='gray', ls=':', lw=1, label='uniform gating (T=0)')
    plt.xscale('symlog', linthresh=1e-2)
    plt.xlabel('temperature T (gating logits scaled by T)')
    plt.ylabel('dataset log-likelihood L(T)')
    plt.title('Gating temperature fit')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / 'temperature_fit.png', dpi=150)
    print(f'Saved curve -> {outdir / "temperature_fit.png"}')

    result = dict(
        T_star=T_star, L_star=L_star, L_uniform=L_uniform, L_T1=L_T1,
        L_hard=L_hard, L_single_expert=L_single,
        per_seq={'T_star': per_seq(L_star), 'uniform': per_seq(L_uniform)},
        n_sequences=N, n_experts=K, n_dropped=dropped,
        weighted=not args.no_weight, length_normalized=args.length_normalize,
        t_max=args.t_max, source=str(args.loglik),
    )
    (outdir / 'temperature_fit.json').write_text(json.dumps(result, indent=2))
    print(f'Saved result -> {outdir / "temperature_fit.json"}')


if __name__ == '__main__':
    main()
