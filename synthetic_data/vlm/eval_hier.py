#!/usr/bin/env python3
"""Evaluate a K=4 hierarchy: hard routing by task, soft mixture within task.

The partition is task-then-random, so the two levels need different treatment:

  * The task level carries signal and is readable from the question, so route
    hard -- an OCR question goes to the OCR group, full stop.
  * The split *within* a task is random, so nothing can predict which member a
    sample "belongs" to. Averaging the members' next-token distributions is the
    only principled use of them; picking one would just throw half the training
    data away.

Reported for contrast:
  single      -- hard task routing, then one arbitrary member per group (the
                 "route and pick" baseline: each expert saw only ~5k examples)
  mixture     -- hard task routing, then a 0.5/0.5 token-wise mixture (the design)
  dense, k2   -- the reference points

    python -m synthetic_data.vlm.eval_hier --split val
"""

import json
import argparse
from pathlib import Path

import torch

from .data import VQADataset, load_records, get_split
from .train import predict, score, fmt_report
from .eval_mixture import load_run, mixture_predict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--features',
                    default='synthetic_data/clip/features_vit_base_patch16.npy')
    R = 'synthetic_data/vlm/runs/sweep'
    ap.add_argument('--ocr-experts', nargs='+',
                    default=[f'{R}/k4__20k__ocr_a', f'{R}/k4__20k__ocr_b'])
    ap.add_argument('--shape-experts', nargs='+',
                    default=[f'{R}/k4__20k__shape_a', f'{R}/k4__20k__shape_b'])
    ap.add_argument('--dense', default=f'{R}/size__20k__dense')
    ap.add_argument('--k2-ocr', default=f'{R}/context__20k__expert_ocr')
    ap.add_argument('--k2-shape', default=f'{R}/context__20k__expert_shape')
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output',
                    default='synthetic_data/vlm/runs/sweep/eval_k4_20k.json')
    args = ap.parse_args()

    device = torch.device(args.device)
    records, feats = load_records(args.annotations, args.features)
    ev = get_split(records, args.split)
    is_ocr = [r['task_type'] == 'ocr' for r in ev]
    idx_ocr = [i for i, o in enumerate(is_ocr) if o]
    idx_shape = [i for i, o in enumerate(is_ocr) if not o]
    print(f'{args.split}: {len(ev)} pairs  ({len(idx_ocr)} ocr, '
          f'{len(idx_shape)} shape)')

    out = {}

    def report(name, preds):
        out[name] = score(ev, preds)
        print('\n' + fmt_report(name, out[name]))
        return 100 * out[name]['exact_match']

    if Path(args.dense, 'best.pt').exists():
        dm, dtok, dst = load_run(args.dense, feats.shape[1], device)
        dense_em = report('DENSE',
                          predict(dm, VQADataset(ev, feats, dtok, dst), device))
    else:
        dense_em = None

    groups = {'ocr': (args.ocr_experts, idx_ocr),
              'shape': (args.shape_experts, idx_shape)}
    members = {g: [load_run(d, feats.shape[1], device) for d in dirs]
               for g, (dirs, _) in groups.items()}
    vocab = members['ocr'][0][1].itos
    for g in members:
        for _, tk, _ in members[g]:
            if tk.itos != vocab:
                raise SystemExit('experts do not share a vocabulary; '
                                 'retrain with --tokenizer')
    print(f'shared vocabulary of {len(vocab)} symbols')

    # each expert alone, on its own group's samples
    for g, (dirs, idx) in groups.items():
        for d, (m, tk, st) in zip(dirs, members[g]):
            sub = [ev[i] for i in idx]
            rep = score(sub, predict(m, VQADataset(sub, feats, tk, st), device))
            out[Path(d).name] = rep
            print(f'  {Path(d).name:<22} alone on its {g} samples '
                  f'(n={len(sub)}): {100 * rep["exact_match"]:.2f}%')

    # hard task routing + first member only
    single = [None] * len(ev)
    for g, (dirs, idx) in groups.items():
        m, tk, st = members[g][0]
        sub = [ev[i] for i in idx]
        for i, p in zip(idx, predict(m, VQADataset(sub, feats, tk, st), device)):
            single[i] = p
    single_em = report('K4 single (task route, one member)', single)

    # hard task routing + 0.5/0.5 mixture within group
    mixed = [None] * len(ev)
    for g, (_, idx) in groups.items():
        sub = [ev[i] for i in idx]
        w = [1.0 / len(members[g])] * len(members[g])
        for i, p in zip(idx, mixture_predict(members[g], w, sub, feats, device)):
            mixed[i] = p
    mix_em = report('K4 mixture (task route, 0.5/0.5 within task)', mixed)

    # K=2 context routing reference
    k2_em = None
    if Path(args.k2_ocr, 'best.pt').exists():
        k2 = [None] * len(ev)
        for run_dir, idx in ((args.k2_ocr, idx_ocr), (args.k2_shape, idx_shape)):
            m, tk, st = load_run(run_dir, feats.shape[1], device)
            sub = [ev[i] for i in idx]
            for i, p in zip(idx, predict(m, VQADataset(sub, feats, tk, st), device)):
                k2[i] = p
        k2_em = report('K2 context routing', k2)

    print('\nsummary (exact match, %s)' % args.split)
    for label, v in (('dense', dense_em), ('K2 context routing', k2_em),
                     ('K4 single (route+pick)', single_em),
                     ('K4 mixture (route+average)', mix_em)):
        if v is None:
            continue
        d = f'  ({v - dense_em:+.2f} vs dense)' if dense_em is not None else ''
        print(f'  {label:<28}{v:6.2f}%{d}')

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
