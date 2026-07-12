#!/usr/bin/env python3
"""
Extract CLIP ViT-B/16 image features for a unique-image JSONL and save a single .npy.

Reads the unique-image JSONL produced by build_unique_image_jsonl.py (one
{"image": <relpath>, "key": <dedup key>} per line), loads each image from
--image-root, runs openai/clip-vit-base-patch16 (get_image_features), and writes:

    <output>.npy            float32 array of shape (N, 512), L2-normalizable features
    <output>_keys.json      list of N dedup keys, aligned row-for-row with the .npy
    <output>_paths.json     list of N relative image paths, aligned row-for-row

The .npy is directly consumable by single_stage_balanced_kmeans.py:
    python clustering/single_stage_balanced_kmeans.py <output>.npy --n-clusters 2

The aligned *_keys.json lets the downstream partition step map any original JSONL
line (via build_unique_image_jsonl.image_key) to its feature row / cluster.

Designed for a single A100: batched GPU inference in fp16, multi-worker image
decoding, and resumable sharded checkpoints so a crash doesn't lose work.

Usage:
    python clustering/extract_clip_features.py \
        --input clustering/unique_images.jsonl \
        --image-root /lambda/nfs/virginia/Decentralized-AR-InternVL/internvl_chat \
        --output clustering/unique_features_vit_base_patch16 \
        --batch-size 512 --num-workers 16
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor

Image.MAX_IMAGE_PIXELS = None  # some infographic/doc images are very large
ImageFile.LOAD_TRUNCATED_IMAGES = True  # some kvqa JPEGs are truncated by a few bytes


class ImageListDataset(Dataset):
    """Yields (index, pixel_values) for each entry; None pixel_values on load failure."""

    def __init__(self, records, image_root, processor):
        self.records = records
        self.image_root = Path(image_root)
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel = self.records[idx]['image']
        try:
            with Image.open(self.image_root / rel) as img:
                img = img.convert('RGB')
                pixel_values = self.processor(images=img, return_tensors='pt')['pixel_values'][0]
            return idx, pixel_values
        except Exception as e:
            if idx < 5 or os.environ.get('CLIP_DEBUG'):
                print(f"  WARN: failed to load {rel}: {e}")
            return idx, None


def collate(batch):
    idxs, pvs = [], []
    for idx, pv in batch:
        if pv is not None:
            idxs.append(idx)
            pvs.append(pv)
    if not pvs:
        return [], None
    return idxs, torch.stack(pvs)


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/16 features to a single .npy")
    parser.add_argument('--input', type=str, default='clustering/unique_images.jsonl',
                        help='Unique-image JSONL from build_unique_image_jsonl.py')
    parser.add_argument('--image-root', type=str,
                        default='/lambda/nfs/virginia/Decentralized-AR-InternVL/internvl_chat',
                        help='Root under which relative "data/..." image paths resolve')
    parser.add_argument('--output', type=str, default='clustering/unique_features_vit_base_patch16',
                        help='Output prefix (writes <output>.npy, <output>_keys.json, <output>_paths.json)')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Run the vision tower in fp16 (default on)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.add_argument('--shard-size', type=int, default=50000,
                        help='Rows per resumable checkpoint shard')
    args = parser.parse_args()

    # Load records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n_total = len(records)
    print(f"Loaded {n_total} unique image records from {args.input}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output.parent / (output.name + '_shards')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load model + processor
    device = torch.device(args.device)
    print(f"Loading {args.model} on {device} (fp16={args.fp16})...")
    processor = CLIPImageProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    if args.fp16:
        model.half()
    feat_dim = model.config.projection_dim

    dataset = ImageListDataset(records, args.image_root, processor)

    # Resumable sharding: process in contiguous shards, skip completed ones.
    shard_bounds = list(range(0, n_total, args.shard_size))
    for shard_start in shard_bounds:
        shard_end = min(shard_start + args.shard_size, n_total)
        shard_file = ckpt_dir / f"shard_{shard_start:09d}_{shard_end:09d}.npy"
        if shard_file.exists():
            print(f"Shard {shard_start}-{shard_end} already done, skipping.")
            continue

        # NaN rows mark failed/missing images; filtered out at the end.
        feats = np.full((shard_end - shard_start, feat_dim), np.nan, dtype=np.float32)

        subset = torch.utils.data.Subset(dataset, list(range(shard_start, shard_end)))
        loader = DataLoader(
            subset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=collate, pin_memory=True,
        )

        for idxs, pixel_values in tqdm(loader, desc=f"shard {shard_start}-{shard_end}"):
            if pixel_values is None:
                continue
            pixel_values = pixel_values.to(device, non_blocking=True)
            if args.fp16:
                pixel_values = pixel_values.half()
            with torch.no_grad():
                # Projected image embedding (== classic CLIPModel.get_image_features).
                # Done explicitly so it's robust across transformers versions whose
                # get_image_features return type has changed.
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                image_embeds = model.visual_projection(vision_outputs.pooler_output)
            out = image_embeds.float().cpu().numpy()
            for j, global_idx in enumerate(idxs):
                feats[global_idx - shard_start] = out[j]

        np.save(shard_file, feats)
        print(f"Saved shard -> {shard_file}")

    # Combine shards in order
    print("\nCombining shards...")
    all_feats = []
    for shard_start in shard_bounds:
        shard_end = min(shard_start + args.shard_size, n_total)
        shard_file = ckpt_dir / f"shard_{shard_start:09d}_{shard_end:09d}.npy"
        all_feats.append(np.load(shard_file))
    features = np.vstack(all_feats)
    assert features.shape[0] == n_total, (features.shape[0], n_total)

    # Filter out failed rows (any NaN) and keep aligned keys/paths
    valid = ~np.isnan(features).any(axis=1)
    n_failed = int((~valid).sum())
    features = features[valid]
    keys = [records[i]['key'] for i in range(n_total) if valid[i]]
    paths = [records[i]['image'] for i in range(n_total) if valid[i]]

    npy_path = str(output) + '.npy'
    np.save(npy_path, features)
    with open(str(output) + '_keys.json', 'w') as f:
        json.dump(keys, f)
    with open(str(output) + '_paths.json', 'w') as f:
        json.dump(paths, f)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Features shape        : {features.shape} ({features.dtype})")
    print(f"Failed/missing images : {n_failed}")
    print(f"Saved features        : {npy_path}")
    print(f"Saved aligned keys    : {output}_keys.json")
    print(f"Saved aligned paths   : {output}_paths.json")
    print(f"\nCluster with:\n  python clustering/single_stage_balanced_kmeans.py {npy_path} --n-clusters 2")


if __name__ == '__main__':
    main()
