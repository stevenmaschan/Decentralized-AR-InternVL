#!/usr/bin/env python3
"""Partition the dataset into two shards at random (control for the α / task splits).

Assignment is per image, by a seeded shuffle, so the two shards are disjoint in
images and carry no structure at all: each is an unbiased ~50% sample of the whole
distribution. Any gain from combining experts trained on these shards is therefore
an *ensembling* effect, not specialisation.

QA pairs are carried over verbatim; images are shared via meta.json "root".

Usage:
    python synthetic_data/split_random.py --output-dir synthetic_data/experts_random
"""

import json
import random
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
    ap.add_argument('--output-dir', default='synthetic_data/experts_random')
    ap.add_argument('--root', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--names', nargs=2, default=['expert_rand_a', 'expert_rand_b'])
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    src_root = args.root if args.root is not None else str(ann_path.parent) + '/'
    out_dir = Path(args.output_dir)

    rows = [json.loads(l) for l in open(ann_path)]
    ids = sorted({r['image_id'] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    side = {i: (0 if k < half else 1) for k, i in enumerate(ids)}

    shards = {n: [] for n in args.names}
    for r in rows:
        shards[args.names[side[r['image_id']]]].append(r)

    meta_by_id = {}
    if Path(args.metadata).exists():
        for line in open(args.metadata):
            m = json.loads(line)
            meta_by_id[m['image_id']] = m

    print(f'{len(rows)} QA pairs from {ann_path}  (seed {args.seed})\n')
    for name in args.names:
        rs = shards[name]
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'annotations.jsonl', 'w') as f:
            for r in rs:
                f.write(json.dumps(r) + '\n')
        img_ids = sorted({r['image_id'] for r in rs})
        if meta_by_id:
            with open(d / 'metadata.jsonl', 'w') as f:
                for i in img_ids:
                    if i in meta_by_id:
                        f.write(json.dumps(meta_by_id[i]) + '\n')
        with open(d / 'meta.json', 'w') as f:
            json.dump({name: {'root': src_root,
                              'annotation': str(d / 'annotations.jsonl'),
                              'data_augment': False, 'repeat_time': 1,
                              'length': len(rs)}}, f, indent=2)
        alphas = [r['alpha'] for r in rs]
        tasks = Counter(r['task_id'] for r in rs)
        n_ocr = sum(1 for r in rs if r['task_type'] == 'ocr')
        print(f'{name}: {len(rs)} QA pairs over {len(img_ids)} images -> {d}')
        print(f'  alpha mean {sum(alphas)/len(alphas):.4f}   '
              f'ocr {n_ocr} ({100*n_ocr/len(rs):.1f}%)   '
              f'splits {dict(Counter(r.get("split") for r in rs))}')
        print('  tasks: ' + '  '.join(f'{TASK_NAME[t]}={tasks[t]}'
                                      for t in sorted(tasks)))
        print()
    total = sum(len(v) for v in shards.values())
    all_ids = [r['id'] for v in shards.values() for r in v]
    assert len(set(all_ids)) == total
    print(f'disjoint and complete: {total} pairs, {len(set(all_ids))} unique ids')


if __name__ == '__main__':
    main()
