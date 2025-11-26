#!/usr/bin/env python3
"""
Convert old format NPZ files (dictionary with image paths as keys) to stacked numpy arrays.

This script:
1. Loads features from old format NPZ (dictionary format)
2. Extracts features in order (from metadata image_paths or sorted keys)
3. Stacks them into a single numpy array
4. Saves the array and corresponding image paths list
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def convert_npz_to_array(npz_file, metadata_file, output_dir):
    """
    Convert NPZ file from dictionary format to stacked array format.
    
    Args:
        npz_file: Path to .npz file (old format with image paths as keys, or new format with 'features' key)
        metadata_file: Path to metadata.json file
        output_dir: Directory to save output files
    """
    print(f"Processing {npz_file}...")
    
    # Load NPZ file
    data = np.load(npz_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check if already in new format
    if 'features' in data:
        print("Already in new format, converting to array format...")
        features_array = data['features']
        
        # Get image paths
        if 'image_paths' in data:
            if isinstance(data['image_paths'], np.ndarray):
                valid_paths = data['image_paths'].tolist()
            else:
                valid_paths = list(data['image_paths'])
        else:
            print("Warning: No image_paths in NPZ, creating placeholder paths")
            valid_paths = [f"image_{i:06d}" for i in range(len(features_array))]
    else:
        # Old format: dictionary with image paths as keys
        print("Old format detected, converting...")
        
        # Get ordered image paths from metadata (preserves original order)
        if 'image_paths' in metadata:
            ordered_paths = metadata['image_paths']
        else:
            # Fallback: use sorted keys
            ordered_paths = sorted(data.keys())
            print(f"Warning: No image_paths in metadata, using sorted keys")
        
        # Extract features in order
        features_list = []
        valid_paths = []
        
        for path in ordered_paths:
            if path in data:
                feature = data[path]
                # Remove batch dimension if present (should be (1, 768))
                if len(feature.shape) > 1 and feature.shape[0] == 1:
                    feature = feature[0]
                # Ensure it's 1D
                if len(feature.shape) > 1:
                    feature = feature.flatten()
                features_list.append(feature)
                valid_paths.append(path)
            else:
                print(f"Warning: Path {path} not found in NPZ file, skipping")
        
        if len(features_list) == 0:
            print(f"Error: No valid features found!")
            return
        
        # Stack features into single array
        features_array = np.stack(features_list)
    
    print(f"Stacked features shape: {features_array.shape}")
    print(f"Number of image paths: {len(valid_paths)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset name from metadata or filename
    dataset_name = metadata.get('dataset_name', Path(npz_file).stem.replace('_clip_features', ''))
    
    # Save stacked features array
    array_file = os.path.join(output_dir, f"{dataset_name}_clip_features_array.npy")
    np.save(array_file, features_array)
    print(f"Saved features array to {array_file}")
    
    # Save image paths in corresponding order as JSON
    paths_file = os.path.join(output_dir, f"{dataset_name}_clip_features_array_paths.json")
    with open(paths_file, 'w') as f:
        json.dump(valid_paths, f, indent=2)
    print(f"Saved image paths to {paths_file}")
    
    # Update metadata
    new_metadata = {
        "dataset_name": dataset_name,
        "model": metadata.get("model", "CLIP ViT-L/14"),
        "input_size": metadata.get("input_size", 336),
        "feature_dim": features_array.shape[1],
        "num_images": len(valid_paths),
        "array_shape": list(features_array.shape),
        "format": "stacked_array",
        "paths_file": os.path.basename(paths_file),
        "paths_format": "json"
    }
    
    metadata_file_new = os.path.join(output_dir, f"{dataset_name}_clip_features_array_metadata.json")
    with open(metadata_file_new, 'w') as f:
        json.dump(new_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file_new}")
    
    print("Done!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert old format NPZ files to stacked numpy arrays"
    )
    parser.add_argument(
        "--npz_file",
        type=str,
        help="Path to .npz file (old format)"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="Path to metadata.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as NPZ file directory)"
    )
    parser.add_argument(
        "--clustering_dir",
        type=str,
        default="/home/maschan/ddfm/InternVL/clustering",
        help="Base clustering directory to process all datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to process (if not provided, processes all in clustering_dir)"
    )
    
    args = parser.parse_args()
    
    # Process single file if provided
    if args.npz_file and args.metadata_file:
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.npz_file)
        convert_npz_to_array(args.npz_file, args.metadata_file, args.output_dir)
        return
    
    # Process all datasets in clustering directory
    if args.dataset:
        datasets = [args.dataset]
    else:
        # Find all datasets with NPZ files (both old and new format)
        clustering_dir = args.clustering_dir
        datasets = []
        for item in os.listdir(clustering_dir):
            dataset_dir = os.path.join(clustering_dir, item)
            if os.path.isdir(dataset_dir):
                npz_file = os.path.join(dataset_dir, f"{item}_clip_features.npz")
                metadata_file = os.path.join(dataset_dir, f"{item}_metadata.json")
                # Check if array file already exists
                array_file = os.path.join(dataset_dir, f"{item}_clip_features_array.npy")
                if os.path.exists(npz_file) and os.path.exists(metadata_file) and not os.path.exists(array_file):
                    datasets.append(item)
        
        print(f"Found {len(datasets)} datasets with old format:")
        for d in datasets:
            print(f"  - {d}")
        print()
    
    # Process each dataset
    for dataset_name in datasets:
        dataset_dir = os.path.join(args.clustering_dir, dataset_name)
        npz_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features.npz")
        metadata_file = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
        
        if not os.path.exists(npz_file):
            print(f"Warning: {npz_file} not found, skipping {dataset_name}")
            continue
        if not os.path.exists(metadata_file):
            print(f"Warning: {metadata_file} not found, skipping {dataset_name}")
            continue
        
        convert_npz_to_array(npz_file, metadata_file, dataset_dir)


if __name__ == "__main__":
    main()

