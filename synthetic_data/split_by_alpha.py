#!/usr/bin/env python3
"""Split the dataset into two expert shards by the generative parameter alpha.

    alpha <  threshold  ->  expert_alpha_lo   (shape-dominated images)
    alpha >= threshold  ->  expert_alpha_hi   (text-dominated images)

This is the *oracle* partition: it uses the ground-truth alpha rather than any
learned routing signal, so it upper-bounds what a router that recovers alpha
could achieve, and gives a reference to compare the CLIP k-means partition against.

The QA pairs are carried over verbatim from the existing annotations.jsonl -- the
questions stay exactly as they were sampled (one per image, OCR with probability
alpha), nothing is regenerated or re-asked. Images are not copied; both shards
point at the same images/ directory via the "root" field of meta.json.

Usage:
    python synthetic_data/split_by_alpha.py \
        --annotations synthetic_data/annotations.jsonl \
        --metadata synthetic_data/metadata.jsonl \
        --output-dir synthetic_data/experts_alpha --threshold 0.5
"""

import json
import argparse
from pathlib import Path
from collections import Counter

TASK_NAME = {1: 'ocr', 2: 'count_by_type', 3: 'count_by_color',
             4: 'name_types', 5: 'count_by_angles'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default='synthetic_data/annotations.jsonl')
    ap.add_argument('--metadata', default='synthetic_data/metadata.jsonl')
    ap.add_argument('--output-dir', default='synthetic_data/experts_alpha')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='alpha < threshold -> lo shard, >= threshold -> hi shard')
    ap.add_argument('--root', default=None,
                    help='"root" for meta.json (default: the source dataset dir)')
    ap.add_argument('--names', nargs=2, default=['expert_alpha_lo', 'expert_alpha_hi'])
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    src_root = args.root if args.root is not None else str(ann_path.parent) + '/'
    out_dir = Path(args.output_dir)

    shards = {name: [] for name in args.names}
    lo_name, hi_name = args.names
    for line in open(ann_path):
        a = json.loads(line)
        shards[lo_name if a['alpha'] < args.threshold else hi_name].append(a)

    # Carry the matching metadata rows so each shard is self-describing.
    meta_by_id = {}
    if Path(args.metadata).exists():
        for line in open(args.metadata):
            m = json.loads(line)
            meta_by_id[m['image_id']] = m

    print(f'{sum(len(v) for v in shards.values())} QA pairs from {ann_path}')
    print(f'threshold: alpha < {args.threshold} -> {lo_name}, '
          f'>= {args.threshold} -> {hi_name}\n')

    for name in args.names:
        rows = shards[name]
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)

        with open(d / 'annotations.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')

        img_ids = sorted({r['image_id'] for r in rows})
        if meta_by_id:
            with open(d / 'metadata.jsonl', 'w') as f:
                for i in img_ids:
                    if i in meta_by_id:
                        f.write(json.dumps(meta_by_id[i]) + '\n')

        with open(d / 'meta.json', 'w') as f:
            json.dump({name: {
                'root': src_root,
                'annotation': str(d / 'annotations.jsonl'),
                'data_augment': False,
                'repeat_time': 1,
                'length': len(rows),
            }}, f, indent=2)

        alphas = [r['alpha'] for r in rows]
        tasks = Counter(r['task_id'] for r in rows)
        n_ocr = sum(1 for r in rows if r['task_type'] == 'ocr')
        print(f'{name}: {len(rows)} QA pairs over {len(img_ids)} images  -> {d}')
        print(f'  alpha  min {min(alphas):.4f}  mean {sum(alphas)/len(alphas):.4f}  '
              f'max {max(alphas):.4f}')
        print(f'  ocr {n_ocr} ({100 * n_ocr / len(rows):.1f}%)  '
              f'shape_reasoning {len(rows) - n_ocr}')
        print('  tasks: ' + '  '.join(
            f'{TASK_NAME[t]}={tasks[t]}' for t in sorted(tasks)))
        print()

    # The two shards must reproduce the original exactly.
    total = sum(len(v) for v in shards.values())
    ids = [r['id'] for v in shards.values() for r in v]
    assert len(set(ids)) == total, 'duplicate ids across shards'
    print(f'disjoint and complete: {total} pairs, {len(set(ids))} unique ids')


if __name__ == '__main__':
    main()
