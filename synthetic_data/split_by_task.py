#!/usr/bin/env python3
"""Split the dataset into two expert shards by *question type* (context routing).

    task_type == 'ocr'              ->  expert_ocr
    task_type == 'shape_reasoning'  ->  expert_shape

The contrast with split_by_alpha.py: that partition is decided by the *image*
(its alpha), so a router must infer it from pixels and every question about an
image goes to the same expert. This one is decided by the *question*, which a
router can read directly off the text -- perfect routing is free -- but it means
an image can be handled by either expert depending on what is asked.

QA pairs are carried over verbatim; images are shared via meta.json "root".

Usage:
    python synthetic_data/split_by_task.py \
        --annotations synthetic_data/annotations.jsonl \
        --output-dir synthetic_data/experts_task
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
    ap.add_argument('--output-dir', default='synthetic_data/experts_task')
    ap.add_argument('--root', default=None)
    ap.add_argument('--names', nargs=2, default=['expert_ocr', 'expert_shape'])
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    src_root = args.root if args.root is not None else str(ann_path.parent) + '/'
    out_dir = Path(args.output_dir)
    ocr_name, shape_name = args.names

    shards = {ocr_name: [], shape_name: []}
    for line in open(ann_path):
        a = json.loads(line)
        shards[ocr_name if a['task_type'] == 'ocr' else shape_name].append(a)

    meta_by_id = {}
    if Path(args.metadata).exists():
        for line in open(args.metadata):
            m = json.loads(line)
            meta_by_id[m['image_id']] = m

    print(f'{sum(len(v) for v in shards.values())} QA pairs from {ann_path}\n')
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
            json.dump({name: {'root': src_root,
                              'annotation': str(d / 'annotations.jsonl'),
                              'data_augment': False, 'repeat_time': 1,
                              'length': len(rows)}}, f, indent=2)
        alphas = [r['alpha'] for r in rows]
        tasks = Counter(r['task_id'] for r in rows)
        by_split = Counter(r.get('split') for r in rows)
        print(f'{name}: {len(rows)} QA pairs over {len(img_ids)} images -> {d}')
        print(f'  alpha mean {sum(alphas)/len(alphas):.4f}  '
              f'splits {dict(by_split)}')
        print('  tasks: ' + '  '.join(f'{TASK_NAME[t]}={tasks[t]}'
                                      for t in sorted(tasks)))
        print()

    total = sum(len(v) for v in shards.values())
    ids = [r['id'] for v in shards.values() for r in v]
    assert len(set(ids)) == total, 'duplicate ids across shards'
    print(f'disjoint and complete: {total} pairs, {len(set(ids))} unique ids')


if __name__ == '__main__':
    main()
