#!/usr/bin/env python3
"""
Stack CLIP features from all datasets into a single consolidated array.

This script:
1. Finds all CLIP feature files for each dataset
2. Loads features from each dataset
3. Stacks them into a single NPY array
4. Creates metadata and mapping files
"""

import json
import os
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict


def find_feature_files(clustering_dir):
    """Find all CLIP feature files for each dataset."""
    datasets = {}
    
    for dataset_dir in os.listdir(clustering_dir):
        dataset_path = os.path.join(clustering_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        # Look for feature files (prioritize array.npy, then npz)
        array_npy = os.path.join(dataset_path, f"{dataset_dir}_clip_features_array.npy")
        npz_file = os.path.join(dataset_path, f"{dataset_dir}_clip_features.npz")
        
        if os.path.exists(array_npy):
            datasets[dataset_dir] = {
                'format': 'npy_array',
                'file': array_npy,
                'paths_file': os.path.join(dataset_path, f"{dataset_dir}_clip_features_array_paths.json"),
            }
        elif os.path.exists(npz_file):
            datasets[dataset_dir] = {
                'format': 'npz',
                'file': npz_file,
                'paths_file': None,
            }
    
    return datasets


def load_features_npy_array(file_path):
    """Load features from .npy array file."""
    features = np.load(file_path, mmap_mode='r')
    return features


def load_features_npz(npz_file):
    """Load features from .npz file."""
    with np.load(npz_file, allow_pickle=True) as npz:
        keys = list(npz.keys())
        # Find features key
        features_key = None
        for key in keys:
            arr = npz[key]
            if arr.dtype.kind == 'f' and len(arr.shape) == 2:
                features_key = key
                break
        
        if features_key is None:
            raise ValueError(f"Could not find features in {npz_file}")
        
        features = npz[features_key]
        return features


def main():
    parser = argparse.ArgumentParser(description="Stack CLIP features from all datasets")
    parser.add_argument(
        "--clustering_dir",
        type=str,
        default="clustering",
        help="Directory containing dataset feature files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clustering/consolidated_all_clip_features.npy",
        help="Output consolidated NPY file"
    )
    parser.add_argument(
        "--output_paths_file",
        type=str,
        default="clustering/consolidated_all_clip_features_paths.txt",
        help="Output paths file (one per line, matching array rows)"
    )
    parser.add_argument(
        "--output_metadata_file",
        type=str,
        default="clustering/consolidated_all_clip_features_metadata.json",
        help="Output metadata JSON file"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STACKING CLIP FEATURES FROM ALL DATASETS")
    print("=" * 80)
    
    # Find all feature files
    print(f"\nFinding feature files in {args.clustering_dir}...")
    datasets = find_feature_files(args.clustering_dir)
    
    print(f"Found {len(datasets)} datasets with feature files:")
    for dataset_name, info in sorted(datasets.items()):
        print(f"  - {dataset_name}: {info['format']}")
    
    # Load features from each dataset
    print(f"\nLoading features from all datasets...")
    all_features_list = []
    all_paths_list = []
    dataset_ranges = {}  # Track start and end indices for each dataset
    total_features = 0
    
    for dataset_name in sorted(datasets.keys()):
        info = datasets[dataset_name]
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load features
            if info['format'] == 'npy_array':
                features = load_features_npy_array(info['file'])
                print(f"  Loaded {len(features):,} features from {info['file']}")
                
                # Load paths if available
                paths = None
                if info['paths_file'] and os.path.exists(info['paths_file']):
                    with open(info['paths_file'], 'r') as f:
                        paths = json.load(f)
                    print(f"  Loaded {len(paths):,} paths")
            else:  # npz
                features = load_features_npz(info['file'])
                print(f"  Loaded {len(features):,} features from {info['file']}")
                
                # Try to extract paths from NPZ
                paths = None
                with np.load(info['file'], allow_pickle=True) as npz:
                    keys = list(npz.keys())
                    for key in keys:
                        arr = npz[key]
                        if arr.dtype.kind in 'USO' or 'path' in key.lower():
                            paths = [str(p) for p in arr]
                            print(f"  Loaded {len(paths):,} paths from NPZ")
                            break
            
            # Record dataset range
            start_idx = total_features
            end_idx = total_features + len(features)
            dataset_ranges[dataset_name] = {
                'start': start_idx,
                'end': end_idx,
                'count': len(features)
            }
            
            # Append features
            all_features_list.append(features)
            
            # Append paths (or placeholder if missing)
            if paths:
                all_paths_list.extend(paths)
            else:
                # Create placeholder paths
                all_paths_list.extend([f"{dataset_name}/unknown_{i}" for i in range(len(features))])
            
            total_features += len(features)
            
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal features to stack: {total_features:,}")
    
    # Stack all features
    print(f"\nStacking features...")
    consolidated_features = np.vstack(all_features_list)
    print(f"Consolidated features shape: {consolidated_features.shape}")
    
    # Save consolidated features
    print(f"\nSaving consolidated features to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, consolidated_features)
    
    # Save paths
    print(f"Saving paths to {args.output_paths_file}...")
    with open(args.output_paths_file, 'w') as f:
        for path in all_paths_list:
            f.write(f"{path}\n")
    
    # Save metadata
    print(f"Saving metadata to {args.output_metadata_file}...")
    metadata = {
        'total_features': int(total_features),
        'feature_dim': int(consolidated_features.shape[1]),
        'datasets': dataset_ranges,
        'num_datasets': len(dataset_ranges),
    }
    
    with open(args.output_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total features: {total_features:,}")
    print(f"Feature dimension: {consolidated_features.shape[1]}")
    print(f"Number of datasets: {len(dataset_ranges)}")
    print(f"\nDataset ranges:")
    for dataset_name, range_info in sorted(dataset_ranges.items()):
        print(f"  {dataset_name:20s}: indices [{range_info['start']:>8,}, {range_info['end']:>8,}) - {range_info['count']:>8,} features")
    print(f"\nOutput file: {args.output_file}")
    print(f"Output shape: {consolidated_features.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()

