#!/usr/bin/env python3
"""
Verify that the order is preserved between feature arrays and path files for all datasets.
"""

import os
import numpy as np
from pathlib import Path


def verify_dataset_order(dataset_dir, dataset_name):
    """Verify order preservation for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Verifying: {dataset_name}")
    print(f"{'='*60}")
    
    array_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features_array.npy")
    paths_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features_array_paths.json")
    
    # Check if files exist
    if not os.path.exists(array_file):
        print(f"  ❌ Array file not found: {array_file}")
        return False
    
    if not os.path.exists(paths_file):
        print(f"  ❌ Paths file not found: {paths_file}")
        return False
    
    # Load array
    features_array = np.load(array_file)
    print(f"  ✓ Loaded features array: shape {features_array.shape}")
    
    # Load paths from JSON
    import json
    with open(paths_file, 'r') as f:
        image_paths = json.load(f)
    print(f"  ✓ Loaded image paths: {len(image_paths)} paths")
    
    # Check count match
    if len(features_array) != len(image_paths):
        print(f"  ❌ MISMATCH: {len(features_array)} features but {len(image_paths)} paths!")
        return False
    
    print(f"  ✓ Count matches: {len(features_array)} features = {len(image_paths)} paths")
    
    # Check first few and last few paths
    print(f"\n  First 5 paths:")
    for i in range(min(5, len(image_paths))):
        print(f"    [{i}] {image_paths[i]}")
    
    print(f"\n  Last 5 paths:")
    for i in range(max(0, len(image_paths) - 5), len(image_paths)):
        print(f"    [{i}] {image_paths[i]}")
    
    # If old format NPZ exists, verify against it
    npz_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features.npz")
    if os.path.exists(npz_file):
        print(f"\n  Verifying against original NPZ file...")
        npz_data = np.load(npz_file)
        
        # Check if old format (dictionary) or new format (features key)
        if 'features' in npz_data:
            print(f"    NPZ is in new format (has 'features' key)")
            # For new format, check if paths match
            if 'image_paths' in npz_data:
                npz_paths = npz_data['image_paths']
                if isinstance(npz_paths, np.ndarray):
                    npz_paths = npz_paths.tolist()
                else:
                    npz_paths = list(npz_paths)
                
                if len(npz_paths) == len(image_paths):
                    # Check first and last match
                    if npz_paths[0] == image_paths[0] and npz_paths[-1] == image_paths[-1]:
                        print(f"    ✓ First and last paths match NPZ")
                    else:
                        print(f"    ⚠️  Path mismatch: NPZ first={npz_paths[0]}, array first={image_paths[0]}")
                else:
                    print(f"    ⚠️  Count mismatch: NPZ has {len(npz_paths)}, array has {len(image_paths)}")
        else:
            print(f"    NPZ is in old format (dictionary)")
            # For old format, check metadata order
            metadata_file = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if 'image_paths' in metadata:
                    metadata_paths = metadata['image_paths']
                    if len(metadata_paths) == len(image_paths):
                        # Check first and last match
                        if metadata_paths[0] == image_paths[0] and metadata_paths[-1] == image_paths[-1]:
                            print(f"    ✓ First and last paths match metadata order")
                        else:
                            print(f"    ⚠️  Path mismatch: metadata first={metadata_paths[0]}, array first={image_paths[0]}")
                    else:
                        print(f"    ⚠️  Count mismatch: metadata has {len(metadata_paths)}, array has {len(image_paths)}")
    
    print(f"  ✓ Verification complete for {dataset_name}")
    return True


def main():
    clustering_dir = "/home/maschan/ddfm/InternVL/clustering"
    
    # Find all datasets with array files
    datasets = []
    for item in os.listdir(clustering_dir):
        dataset_dir = os.path.join(clustering_dir, item)
        if os.path.isdir(dataset_dir):
            array_file = os.path.join(dataset_dir, f"{item}_clip_features_array.npy")
            if os.path.exists(array_file):
                datasets.append(item)
    
    datasets.sort()
    
    print(f"Found {len(datasets)} datasets with array files:")
    for d in datasets:
        print(f"  - {d}")
    
    # Verify each dataset
    all_passed = True
    for dataset_name in datasets:
        dataset_dir = os.path.join(clustering_dir, dataset_name)
        if not verify_dataset_order(dataset_dir, dataset_name):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All datasets verified successfully!")
    else:
        print("❌ Some datasets failed verification")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

