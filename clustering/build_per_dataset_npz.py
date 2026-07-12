#!/usr/bin/env python3
"""
Materialize the per-dataset CLIP-feature NPZ files that
partition_jsonl_by_balanced_kmeans.py expects, from the single global unique
feature array produced by extract_clip_features.py.

No CLIP recompute: features are looked up from unique_features_*.npy via the
same image_key() canonicalization used to build the unique set. Each NPZ stores
`image_paths` exactly as they appear in that dataset's JSONL, so the partition
script matches lines by exact path.

Output layout matches get_npz_file_path(dataset, data_root):
    <data-root>/<dataset_dir>/clip_features/<dataset>_clip_base-patch16_features.npz

Usage:
    python clustering/build_per_dataset_npz.py \
        --dense-dir /lambda/nfs/virginia/internvl-data/dense \
        --features-prefix clustering/unique_features_vit_base_patch16 \
        --data-root clustering/per_dataset_npz
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_unique_image_jsonl import image_key
from partition_jsonl_by_balanced_kmeans import get_npz_file_path


def main():
    parser = argparse.ArgumentParser(description="Build per-dataset NPZ files from unique features")
    parser.add_argument('--dense-dir', type=str, default='/lambda/nfs/virginia/internvl-data/dense')
    parser.add_argument('--features-prefix', type=str,
                        default='clustering/unique_features_vit_base_patch16')
    parser.add_argument('--data-root', type=str, default='clustering/per_dataset_npz')
    parser.add_argument('--feature-type', type=str, default='base-patch16')
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir)
    data_root = Path(args.data_root)

    print("Loading unique features...")
    feats = np.load(args.features_prefix + '.npy')
    keys = json.load(open(args.features_prefix + '_keys.json'))
    key_to_idx = {k: i for i, k in enumerate(keys)}
    print(f"  {feats.shape[0]} unique features, dim {feats.shape[1]}")

    total_missing = 0
    for jsonl_file in sorted(dense_dir.glob('*.jsonl')):
        dataset = jsonl_file.stem
        # unique raw image paths in this dataset (first occurrence order)
        seen = set()
        paths = []
        for line in open(jsonl_file):
            line = line.strip()
            if not line:
                continue
            try:
                img = json.loads(line).get('image')
            except json.JSONDecodeError:
                continue
            if not img:
                continue
            img = img.strip().replace('\\', '/')
            if img not in seen:
                seen.add(img)
                paths.append(img)

        idxs, kept_paths, missing = [], [], 0
        for p in paths:
            idx = key_to_idx.get(image_key(p))
            if idx is None:
                missing += 1
                continue
            idxs.append(idx)
            kept_paths.append(p)
        total_missing += missing

        ds_feats = feats[idxs]
        npz_file = get_npz_file_path(dataset, data_root, feature_type=args.feature_type)
        npz_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_file, features=ds_feats,
                 image_paths=np.array(kept_paths, dtype=object))
        print(f"  {dataset}: {len(kept_paths)} imgs"
              f"{f' ({missing} missing)' if missing else ''} -> {npz_file}")

    print(f"\nDone. Total missing feature lookups: {total_missing}")


if __name__ == '__main__':
    main()
