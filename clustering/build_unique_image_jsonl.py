#!/usr/bin/env python3
"""
Build a single JSONL of UNIQUE image paths across all dense dataset JSONLs.

Many lines (within and across the JSONLs in ~/virginia/internvl-data/dense) refer
to the same underlying image, so computing CLIP features per line would be wasteful.
This script collapses them to a unique set so features are computed once per image.

Two sources of duplication are handled:

1. Exact-path duplicates
   - The same relative path repeated across QA lines (e.g. many vqav2 questions per
     image) and across datasets (textvqa and textcaps both use data/textvqa/train_images).

2. COCO cross-split duplicates
   - aokvqa references data/coco/train2017/<id>.jpg while refcoco*/vqav2 reference
     data/coco/train2014/COCO_train2014_<id>.jpg. COCO 2014 and 2017 share the same
     underlying images keyed by numeric image id, so train2017/train2014/val2014 copies
     of the same id are byte-identical and collapse to one entry.

The canonical image path chosen for each unique image is one that actually exists on
disk under --image-root (preferring train2014 -> val2014 -> train2017 for COCO).

Output: one JSON object per line with:
    {"image": "<canonical relative path>", "key": "<dedup key>"}

`image_key()` is importable so downstream steps (feature NPZ build, cluster partition)
can map any original JSONL path to the same key and thus the same unique image.

Usage:
    python clustering/build_unique_image_jsonl.py \
        --dense-dir /lambda/nfs/virginia/internvl-data/dense \
        --image-root /lambda/nfs/virginia/Decentralized-AR-InternVL/internvl_chat \
        --output clustering/unique_images.jsonl
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Matches a trailing numeric image id before the extension, e.g.
#   000000000074.jpg  ->  74
#   COCO_train2014_000000098304.jpg  ->  98304
_COCO_ID_RE = re.compile(r'(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)


def image_key(image_rel):
    """Canonical dedup key for an image path as it appears in a dense JSONL.

    COCO images (any split) collapse to 'coco:<int id>'. Everything else keys on
    its exact relative path. Keep this in sync with any downstream partition step.
    """
    parts = image_rel.split('/')
    if len(parts) >= 2 and parts[0] == 'data' and parts[1] == 'coco':
        m = _COCO_ID_RE.search(os.path.basename(image_rel))
        if m:
            return f'coco:{int(m.group(1))}'
    return f'path:{image_rel}'


def resolve_canonical_path(image_rel, key, image_root):
    """Return a relative path that exists on disk for this unique image, or None."""
    if key.startswith('coco:'):
        cid = int(key.split(':', 1)[1])
        candidates = [
            f'data/coco/train2014/COCO_train2014_{cid:012d}.jpg',
            f'data/coco/val2014/COCO_val2014_{cid:012d}.jpg',
            f'data/coco/train2017/{cid:012d}.jpg',
        ]
        for cand in candidates:
            if (image_root / cand).is_file():
                return cand
        return None
    # Non-COCO: the path itself is canonical; keep it if it exists.
    if (image_root / image_rel).is_file():
        return image_rel
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build a single JSONL of unique image paths across dense JSONLs"
    )
    parser.add_argument(
        '--dense-dir', type=str,
        default='/lambda/nfs/virginia/internvl-data/dense',
        help='Directory containing the dense dataset JSONL files',
    )
    parser.add_argument(
        '--image-root', type=str,
        default='/lambda/nfs/virginia/Decentralized-AR-InternVL/internvl_chat',
        help='Root under which the relative "data/..." image paths resolve',
    )
    parser.add_argument(
        '--output', type=str,
        default='clustering/unique_images.jsonl',
        help='Output JSONL path',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Specific dataset stems to process (default: all *.jsonl in dense-dir)',
    )
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir)
    image_root = Path(args.image_root)
    output_path = Path(args.output)

    if args.datasets:
        jsonl_files = [dense_dir / f'{name}.jsonl' for name in args.datasets]
    else:
        # glob('*.jsonl') excludes *.jsonl.backup and dataset_mixture.json
        jsonl_files = sorted(dense_dir.glob('*.jsonl'))

    print(f"Found {len(jsonl_files)} JSONL files in {dense_dir}")

    # key -> canonical relative path (first resolved wins)
    key_to_path = {}
    # key -> set of source dataset stems (for reporting)
    key_sources = defaultdict(set)
    # stats
    total_lines = 0
    per_dataset_lines = {}
    missing_examples = []
    missing_count = 0
    no_image = 0

    for jsonl_file in jsonl_files:
        if not jsonl_file.exists():
            print(f"  WARNING: not found, skipping: {jsonl_file}")
            continue
        stem = jsonl_file.stem
        n = 0
        with open(jsonl_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                total_lines += 1
                image_rel = entry.get('image')
                if not image_rel:
                    no_image += 1
                    continue
                image_rel = image_rel.strip().replace('\\', '/')
                key = image_key(image_rel)
                key_sources[key].add(stem)
                if key in key_to_path:
                    continue
                canonical = resolve_canonical_path(image_rel, key, image_root)
                if canonical is None:
                    missing_count += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(image_rel)
                    # Store as unresolved sentinel so we don't retry every line
                    key_to_path[key] = None
                    continue
                key_to_path[key] = canonical
        per_dataset_lines[stem] = n
        print(f"  {stem}: {n} lines")

    # Assemble unique entries (drop unresolved)
    unique = [(k, p) for k, p in key_to_path.items() if p is not None]
    unique.sort(key=lambda kp: kp[1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as out:
        for key, path in unique:
            out.write(json.dumps({'image': path, 'key': key}) + '\n')

    # Report
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total JSONL lines scanned : {total_lines}")
    print(f"Lines with no image       : {no_image}")
    print(f"Distinct image keys       : {len(key_to_path)}")
    print(f"Unique images written     : {len(unique)}")
    print(f"Unresolved (missing file) : {missing_count}")
    if total_lines:
        print(f"Dedup ratio               : {len(unique)/total_lines:.3f} "
              f"({total_lines} -> {len(unique)})")

    coco_keys = sum(1 for k in key_to_path if k.startswith('coco:'))
    print(f"COCO unique image ids      : {coco_keys}")

    # Images shared across >1 dataset
    shared = sum(1 for k, s in key_sources.items() if len(s) > 1)
    print(f"Images shared by >1 dataset: {shared}")

    if missing_examples:
        print("\nExamples of unresolved image paths (not found on disk):")
        for ex in missing_examples:
            print(f"    {ex}")

    print(f"\nWrote unique image JSONL to: {output_path}")


if __name__ == '__main__':
    main()
