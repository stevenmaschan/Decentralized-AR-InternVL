#!/usr/bin/env python3
"""
Combine CLIP features from multiple NPZ files into a single numpy array.

This script loads features from all NPZ files matching a feature type pattern
and concatenates them into a single numpy array.

Usage:
    python combine_vit_large_features.py \
        --feature-type base-patch16 \
        --output-file clustering/unique_features_vit_base_patch16.npy
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def combine_features_from_npz_files(data_root, feature_type='large-patch14-336', use_unique=False):
    """
    Combine features from all NPZ files matching the feature type.
    
    Args:
        data_root: Root directory containing dataset directories
        feature_type: Feature type string (e.g., 'large-patch14-336')
        use_unique: If True, prefer unique NPZ files when available
    
    Returns:
        combined_features: numpy array of shape (N, D)
    """
    data_dir = Path(data_root)
    all_features = []
    
    # Find all NPZ files matching the pattern
    if use_unique:
        pattern = f'*unique*clip_{feature_type}_features.npz'
        npz_files = sorted(list(data_dir.rglob(pattern)))
        # If no unique files found, fall back to regular files
        if not npz_files:
            pattern = f'*clip_{feature_type}_features.npz'
            npz_files = sorted(list(data_dir.rglob(pattern)))
    else:
        pattern = f'*clip_{feature_type}_features.npz'
        npz_files = sorted(list(data_dir.rglob(pattern)))
        # Exclude unique files if we're not using them
        npz_files = [f for f in npz_files if 'unique' not in f.name]
    
    print(f"Found {len(npz_files)} NPZ files")
    
    for npz_file in tqdm(npz_files, desc="Loading features"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            if 'features' in data:
                features = data['features']
            elif 'arr_0' in data:
                features = data['arr_0']
            else:
                print(f"Warning: Could not find features in {npz_file}")
                continue
            
            print(f"  {npz_file.name}: {features.shape}")
            all_features.append(features)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features loaded from NPZ files")
    
    # Combine features
    print("\nCombining features...")
    combined_features = np.vstack(all_features)
    print(f"Combined features shape: {combined_features.shape}")
    
    return combined_features


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine CLIP features from NPZ files")
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root directory containing dataset directories (default: data)'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='large-patch14-336',
        help='Feature type string (default: large-patch14-336)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: auto-generated from feature-type)'
    )
    parser.add_argument(
        '--use-unique',
        action='store_true',
        help='Use unique NPZ files when available'
    )
    
    args = parser.parse_args()
    
    # Combine features
    combined_features = combine_features_from_npz_files(
        args.data_root,
        feature_type=args.feature_type,
        use_unique=args.use_unique
    )
    
    # Determine output file
    if args.output_file is None:
        # Auto-generate filename from feature type
        feature_name = args.feature_type.replace('-', '_')
        output_file = Path('clustering') / f'unique_features_{feature_name}.npy'
    else:
        output_file = Path(args.output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save combined features
    np.save(output_file, combined_features)
    print(f"\nSaved combined features to {output_file}")






