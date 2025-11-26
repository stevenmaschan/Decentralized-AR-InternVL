#!/usr/bin/env python3
"""
Extract CLIP features for ALLaVA sampled dataset from existing ALLaVA features.

This script:
1. Loads ALLaVA sampled JSONL file
2. Loads existing ALLaVA CLIP features
3. Maps sampled images to their features
4. Creates NPZ file with sampled features in the same order
"""

import json
import os
import numpy as np
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for ALLaVA sample from existing features")
    parser.add_argument(
        "--sampled_jsonl",
        type=str,
        default="/media/data2/maschan/internvl/data/allava/allava_instruct_train_sampled.jsonl",
        help="Path to ALLaVA sampled JSONL file"
    )
    parser.add_argument(
        "--allava_features_file",
        type=str,
        default="clustering/allava/allava_clip_features_array.npy",
        help="Path to existing ALLaVA features array"
    )
    parser.add_argument(
        "--allava_paths_file",
        type=str,
        default="clustering/allava/allava_clip_features_array_paths.json",
        help="Path to existing ALLaVA paths JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clustering/allava/allava_sampled_clip_features.npz",
        help="Output NPZ file path"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXTRACTING CLIP FEATURES FOR ALLaVA SAMPLE FROM EXISTING FEATURES")
    print("=" * 80)
    
    # Load sampled JSONL
    print(f"\nLoading sampled JSONL from {args.sampled_jsonl}...")
    sampled_entries = []
    sampled_image_paths = []
    
    with open(args.sampled_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                sampled_entries.append(entry)
                image_path = entry.get('image', '')
                if image_path:
                    sampled_image_paths.append(image_path)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(sampled_entries):,} sampled entries")
    print(f"Found {len(sampled_image_paths):,} image paths")
    
    # Load existing ALLaVA features
    print(f"\nLoading existing ALLaVA features from {args.allava_features_file}...")
    all_features = np.load(args.allava_features_file, mmap_mode='r')
    print(f"Features shape: {all_features.shape}")
    
    # Load existing ALLaVA paths
    print(f"\nLoading existing ALLaVA paths from {args.allava_paths_file}...")
    with open(args.allava_paths_file, 'r') as f:
        all_paths = json.load(f)
    print(f"Found {len(all_paths):,} paths in feature file")
    
    # Create path to index mapping
    print("\nCreating path to index mapping...")
    path_to_index = {}
    for idx, path in enumerate(tqdm(all_paths, desc="Building path mapping")):
        # Normalize path (might be just filename or full path)
        # Store both variations
        filename = os.path.basename(path)
        path_to_index[path] = idx
        path_to_index[filename] = idx
        # Also try with data/allava/images/ prefix
        if not path.startswith('data/'):
            full_path = f"data/allava/images/{path}"
            path_to_index[full_path] = idx
    
    print(f"Created mapping for {len(path_to_index):,} path variations")
    
    # Extract features for sampled images (preserve order)
    print(f"\nExtracting features for {len(sampled_image_paths):,} sampled images...")
    sampled_features = []
    sampled_paths = []
    matched_count = 0
    unmatched_count = 0
    
    for image_path in tqdm(sampled_image_paths, desc="Extracting sampled features"):
        # Try to find index for this path
        idx = None
        
        # Try exact match
        if image_path in path_to_index:
            idx = path_to_index[image_path]
        else:
            # Try with just filename
            filename = os.path.basename(image_path)
            if filename in path_to_index:
                idx = path_to_index[filename]
        
        if idx is not None:
            sampled_features.append(all_features[idx])
            sampled_paths.append(image_path)
            matched_count += 1
        else:
            print(f"Warning: Could not find features for {image_path}, skipping...")
            unmatched_count += 1
            # Skip this image (don't append zero features)
    
    # Stack features
    print(f"\nStacking features...")
    sampled_features_array = np.array(sampled_features)
    print(f"Sampled features shape: {sampled_features_array.shape}")
    
    # Convert paths to numpy array
    sampled_paths_array = np.array(sampled_paths, dtype=object)
    
    # Save as NPZ
    print(f"\nSaving to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    np.savez(
        args.output_file,
        features=sampled_features_array,
        image_paths=sampled_paths_array,
    )
    
    # Save metadata
    metadata = {
        'num_samples': len(sampled_features_array),
        'feature_dim': sampled_features_array.shape[1],
        'matched_count': matched_count,
        'unmatched_count': unmatched_count,
        'input_file': args.sampled_jsonl,
        'source_features_file': args.allava_features_file,
    }
    
    metadata_file = args.output_file.replace('.npz', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sampled entries: {len(sampled_entries):,}")
    print(f"Matched features: {matched_count:,}")
    print(f"Unmatched features: {unmatched_count:,}")
    print(f"Features shape: {sampled_features_array.shape}")
    print(f"Output file: {args.output_file}")
    print(f"Metadata file: {metadata_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

