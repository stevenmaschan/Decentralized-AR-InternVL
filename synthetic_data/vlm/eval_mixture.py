#!/usr/bin/env python3
"""Token-wise soft mixture of several experts, scored against dense and each member.

At every decoding step each model produces a next-character distribution over the
shared vocabulary; those are averaged with fixed weights (0.5/0.5 by default) and
the argmax of the mixture is emitted and fed back to all models. This is a mixture
of *distributions*, not of logits -- the two differ whenever the members disagree,
and averaging probabilities is the one that corresponds to "either expert may be
right" rather than to a product of experts.

Every member must share the vocabulary (train with --tokenizer), since the mixture
sums coordinate-wise over the index space.

    python -m synthetic_data.vlm.eval_mixture --split val \
        --experts runs/sweep/random__p10__expert_rand_a \
                  runs/sweep/random__p10__expert_rand_b
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .data import (CharTokenizer, FeatureStats, VQADataset, load_records,
                   get_split, PAD_ID, EOS_ID)
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


@torch.no_grad()
def mixture_predict(members, weights, records, feats, device, batch_size=256,
                    max_new_tokens=64):
    """Greedy decode from the weighted average of the members' next-token
    distributions. Prompts are bucketed by length so no padding is needed."""
    tok = members[0][1]
    buckets = defaultdict(list)
    for i, r in enumerate(records):
        buckets[len(r['question'])].append(i)

    preds = [None] * len(records)
    for _, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            rows = [records[i]['row'] for i in chunk]
            # Each member standardises the CLIP features with its own train stats.
            feat_per_member = [
                torch.from_numpy(np.ascontiguousarray(st.apply(feats[rows])))
                .float().to(device)
                for _, _, st in members]
            ids = torch.tensor(
                [[1] + tok.encode(records[i]['question']) + [2] for i in chunk],
                dtype=torch.long, device=device)

            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            out = [[] for _ in chunk]
            for _ in range(max_new_tokens):
                mix = None
                for (model, _, _), fv, w in zip(members, feat_per_member, weights):
                    h = model.backbone(fv, ids[:, -model.max_len:])
                    p = F.softmax(model.head(h[:, -1]).float(), dim=-1)
                    mix = w * p if mix is None else mix + w * p
                nxt = mix.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
                for j, t in enumerate(nxt.tolist()):
                    if not done[j] and t != EOS_ID:
                        out[j].append(t)
                done = done | (nxt == EOS_ID)
                if bool(done.all()):
                    break
                ids = torch.cat([ids, nxt[:, None]], dim=1)
            for i, seq in zip(chunk, out):
                preds[i] = tok.decode(seq)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    ap.add_argument('--experts', nargs='+', required=True)
    ap.add_argument('--weights', type=float, nargs='*', default=None,
                    help='mixture weights (default: uniform)')
    ap.add_argument('--dense', default=None,
                    help='optional dense run to compare against')
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    records, feats = load_records(args.annotations, args.features)
    evalset = get_split(records, args.split)
    print(f'{args.split} set: {len(evalset)} QA pairs')

    members = [load_run(d, feats.shape[1], device) for d in args.experts]
    vocabs = [m[1].itos for m in members]
    if any(v != vocabs[0] for v in vocabs):
        raise SystemExit('members do not share a vocabulary; retrain with --tokenizer')
    print(f'{len(members)} members, shared vocabulary of {len(vocabs[0])} symbols')

    w = args.weights or [1.0 / len(members)] * len(members)
    if len(w) != len(members):
        raise SystemExit('--weights must match the number of experts')
    tot = sum(w)
    w = [x / tot for x in w]
    print(f'mixture weights: {w}')

    out = {'weights': w, 'experts': args.experts}

    if args.dense:
        dm, dtok, dst = load_run(args.dense, feats.shape[1], device)
        out['dense'] = score(evalset, predict(dm, VQADataset(evalset, feats, dtok, dst),
                                              device))
        print('\n' + fmt_report('DENSE', out['dense']))

    for d, (m, tk, st) in zip(args.experts, members):
        name = Path(d).name
        out[name] = score(evalset, predict(m, VQADataset(evalset, feats, tk, st), device))
        print('\n' + fmt_report(f'{name} alone (full {args.split})', out[name]))

    out['mixture'] = score(evalset, mixture_predict(members, w, evalset, feats, device))
    print('\n' + fmt_report(f'MIXTURE ({" / ".join(f"{x:.2f}" for x in w)})',
                            out['mixture']))

    print()
    base = 100 * out['dense']['exact_match'] if args.dense else None
    mix = 100 * out['mixture']['exact_match']
    best = max(100 * out[Path(d).name]['exact_match'] for d in args.experts)
    print(f'best single expert {best:.2f}%   mixture {mix:.2f}%   '
          f'delta {mix - best:+.2f}')
    if base is not None:
        print(f'dense              {base:.2f}%   mixture {mix:.2f}%   '
              f'delta {mix - base:+.2f}')

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
