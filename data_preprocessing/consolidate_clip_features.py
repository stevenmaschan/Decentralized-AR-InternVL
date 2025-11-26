#!/usr/bin/env python3
"""
Consolidate CLIP features from all datasets into a single array for unique images.

This script:
1. Loads CLIP features from all datasets in the clustering directory
2. Maps features to image paths
3. Maps paths to unique image identities (from unique_image_paths_normalized.txt)
4. Creates a single consolidated npy array where each row corresponds to a unique image identity
"""

import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def extract_image_identity(image_path):
    """
    Extract a normalized identity from an image path.
    Must match the logic used in add_allava_to_unique_images.py
    """
    import re
    
    # Remove data/ prefix if present
    path = image_path.replace('data/', '')
    
    # Extract filename without extension
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Handle COCO images: COCO_train2014_000000000009.jpg -> coco_000000000009
    coco_match = re.match(r'COCO_(train|val)(\d{4})_(\d+)$', name_without_ext)
    if coco_match:
        img_id = coco_match.group(3)
        return f"coco_{img_id}"
    
    # Handle COCO without prefix: 000000000009.jpg -> coco_000000000009
    if re.match(r'^\d{12}$', name_without_ext):
        return f"coco_{name_without_ext}"
    
    # Handle TextCaps/TextVQA: filename is the identity
    if 'textcaps' in path.lower() or 'textvqa' in path.lower():
        return f"textcaps_textvqa_{filename}"
    
    # For other images, use dataset name + filename as identity
    # Extract dataset name from path
    parts = path.split('/')
    if len(parts) >= 2:
        dataset = parts[0]
        return f"{dataset}_{filename}"
    
    return f"unknown_{filename}"


def normalize_path(path, dataset_name, images_dir=None):
    """
    Normalize a path to match the format in unique_image_paths.txt.
    Paths in feature files might be just filenames, relative paths, or absolute paths.
    """
    # If already a full path starting with data/, return as-is
    if path.startswith('data/'):
        return path
    
    # If absolute path, extract relative part from data/
    if os.path.isabs(path):
        if '/data/' in path:
            idx = path.find('/data/')
            return path[idx+1:]  # +1 to skip leading /
    
    # Handle just filename - need to construct full path
    filename = os.path.basename(path)
    
    # Map dataset names to expected path prefixes
    dataset_path_map = {
        'ai2d': 'data/ai2diagram/images/',
        'docvqa': 'data/docvqa/images/',
        'chartqa': 'data/chartqa/ChartQA Dataset/train/png/',
        'kvqa': 'data/kvqa/raw/KVQAimgs/',
        'gqa': 'data/gqa/images/',
        'infovqa': 'data/infographicsvqa/infographicsvqa_images/',
        'vqav2': 'data/vqav2/train2014/',
        'aokvqa': 'data/coco/train2017/',
        'textcaps': 'data/textcaps/train_images/',
        'textvqa': 'data/textvqa/train_images/',
        'refcoco': 'data/coco/train2014/',
        'refcoco+': 'data/coco/train2014/',
        'refcocog': 'data/coco/train2014/',
        'scienceqa_image': 'data/scienceqa/images/train/',
        'sharegpt4o': 'data/sharegpt4o/images/',
        'allava': 'data/allava/images/',
    }
    
    # Special handling for some datasets
    if dataset_name in dataset_path_map:
        base_path = dataset_path_map[dataset_name]
        # For scienceqa, path might include subdirectory
        if dataset_name == 'scienceqa_image' and '/' in path and not path.startswith('data/'):
            # Path might be like "pid/image.jpg"
            return f"data/scienceqa/images/train/{path}"
        return base_path + filename
    
    # Fallback: use images_dir if provided
    if images_dir:
        if images_dir.startswith('data/'):
            return f"{images_dir}/{filename}"
        elif '/data/' in images_dir:
            idx = images_dir.find('/data/')
            return f"{images_dir[idx+1:]}/{filename}"
    
    # Last resort: construct from dataset name
    return f"data/{dataset_name}/images/{filename}"


def load_features_from_npy(feature_file, paths_file):
    """Load features from .npy file with separate paths JSON file."""
    features = np.load(feature_file)
    
    with open(paths_file, 'r') as f:
        paths = json.load(f)
    
    return features, paths


def load_features_from_npz(npz_file, dataset_name, metadata=None):
    """Load features from .npz file (may contain features and paths)."""
    npz = np.load(npz_file)
    
    # Check what keys are available
    keys = list(npz.keys())
    
    # Try to find features and paths
    features_key = None
    paths_key = None
    
    for key in keys:
        arr = npz[key]
        # Features are numeric arrays
        if arr.dtype.kind in 'f' and len(arr.shape) == 2:
            features_key = key
        # Paths are string arrays
        elif arr.dtype.kind in 'US' or 'path' in key.lower():
            paths_key = key
    
    if features_key is None:
        raise ValueError(f"Could not find features in {npz_file}")
    
    features = npz[features_key]
    
    # Extract paths
    if paths_key:
        paths_array = npz[paths_key]
        # Convert numpy string array to list of strings
        if paths_array.dtype.kind in 'US':
            paths = [str(p) for p in paths_array]
        else:
            paths = paths_array.tolist()
    else:
        # No paths in npz, might need to extract from metadata or use indices
        print(f"Warning: No paths found in {npz_file}, will need to handle separately")
        paths = None
    
    return features, paths


def find_all_feature_files(clustering_dir):
    """Find all CLIP feature files in the clustering directory."""
    feature_files = []
    
    for dataset_dir in os.listdir(clustering_dir):
        dataset_path = os.path.join(clustering_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        # Look for feature files
        npy_file = os.path.join(dataset_path, f"{dataset_dir}_clip_features_array.npy")
        npz_file = os.path.join(dataset_path, f"{dataset_dir}_clip_features.npz")
        paths_json = os.path.join(dataset_path, f"{dataset_dir}_clip_features_array_paths.json")
        metadata_json = os.path.join(dataset_path, f"{dataset_dir}_metadata.json")
        array_metadata_json = os.path.join(dataset_path, f"{dataset_dir}_clip_features_array_metadata.json")
        
        # Determine which files exist and what format
        if os.path.exists(npy_file) and os.path.exists(paths_json):
            feature_files.append({
                'dataset': dataset_dir,
                'format': 'npy',
                'feature_file': npy_file,
                'paths_file': paths_json,
                'metadata_file': array_metadata_json if os.path.exists(array_metadata_json) else None,
            })
        elif os.path.exists(npz_file):
            feature_files.append({
                'dataset': dataset_dir,
                'format': 'npz',
                'feature_file': npz_file,
                'paths_file': None,
                'metadata_file': metadata_json if os.path.exists(metadata_json) else None,
            })
    
    return feature_files


def load_unique_image_identities(identities_file):
    """Load unique image identities from file."""
    identities = []
    with open(identities_file, 'r') as f:
        for line in f:
            identity = line.strip()
            if identity:
                identities.append(identity)
    return sorted(identities)


def load_unique_image_paths(paths_file):
    """Load unique image paths and create identity to path mapping."""
    paths = []
    identity_to_path = {}
    
    with open(paths_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                paths.append(path)
                identity = extract_image_identity(path)
                # Map identity to path (for now, just use first path for each identity)
                # We'll handle duplicates later
                if identity not in identity_to_path:
                    identity_to_path[identity] = path
    
    return paths, identity_to_path


def main():
    clustering_dir = "clustering"
    unique_paths_file = "/media/data2/maschan/internvl/data/unique_image_paths.txt"
    unique_identities_file = "/media/data2/maschan/internvl/data/unique_image_paths_normalized.txt"
    output_file = "clustering/consolidated_clip_features.npy"
    output_paths_file = "clustering/consolidated_clip_features_paths.txt"
    output_metadata_file = "clustering/consolidated_clip_features_metadata.json"
    
    print("=" * 80)
    print("CONSOLIDATING CLIP FEATURES FOR UNIQUE IMAGES")
    print("=" * 80)
    
    # Load unique image identities
    print("\nLoading unique image identities...")
    unique_identities = load_unique_image_identities(unique_identities_file)
    print(f"Found {len(unique_identities):,} unique image identities")
    
    # Create identity to index mapping
    identity_to_index = {identity: idx for idx, identity in enumerate(unique_identities)}
    
    # Load unique paths and create path to identity mapping
    print("\nLoading unique image paths...")
    unique_paths, identity_to_path = load_unique_image_paths(unique_paths_file)
    unique_paths_set = set(unique_paths)
    print(f"Found {len(unique_paths):,} unique image paths")
    print(f"Found {len(identity_to_path):,} unique identities from paths")
    
    # Create path to identity mapping (handle duplicates)
    path_to_identity = {}
    for path in unique_paths:
        identity = extract_image_identity(path)
        path_to_identity[path] = identity
    
    print(f"Mapped {len(path_to_identity):,} paths to identities")
    
    # Find all feature files
    print("\nFinding all feature files...")
    feature_files = find_all_feature_files(clustering_dir)
    print(f"Found {len(feature_files)} datasets with feature files:")
    for ff in feature_files:
        print(f"  - {ff['dataset']}: {ff['format']}")
    
    # Initialize consolidated features array
    num_identities = len(unique_identities)
    feature_dim = 768  # CLIP ViT-L/14 feature dimension
    consolidated_features = np.zeros((num_identities, feature_dim), dtype=np.float32)
    identity_has_features = np.zeros(num_identities, dtype=bool)
    
    # Track which paths we've seen
    path_to_feature = {}
    
    # Load features from each dataset
    print("\nLoading features from all datasets...")
    total_features_loaded = 0
    
    for ff in tqdm(feature_files, desc="Processing datasets"):
        dataset_name = ff['dataset']
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load features and paths
            if ff['format'] == 'npy':
                features, paths = load_features_from_npy(ff['feature_file'], ff['paths_file'])
            else:  # npz
                # Load metadata if available to get images_dir
                metadata = None
                if ff['metadata_file'] and os.path.exists(ff['metadata_file']):
                    with open(ff['metadata_file'], 'r') as f:
                        metadata = json.load(f)
                
                features, paths = load_features_from_npz(ff['feature_file'], dataset_name, metadata)
            
            print(f"  Loaded {len(features):,} features (shape: {features.shape})")
            
            # Normalize paths and map to identities
            if paths is None:
                print(f"  Warning: No paths found for {dataset_name}, skipping")
                continue
            
            print(f"  Mapping {len(paths):,} paths to identities...")
            mapped_count = 0
            
            for i, path in enumerate(tqdm(paths, desc=f"  Mapping {dataset_name}", leave=False)):
                # Normalize path
                normalized_path = normalize_path(path, dataset_name, 
                                                 metadata.get('images_dir') if metadata else None)
                
                # Check if this path is in our unique paths
                if normalized_path in unique_paths_set:
                    # Get identity for this path
                    identity = path_to_identity.get(normalized_path)
                    if identity is None:
                        # Calculate identity if not in mapping
                        identity = extract_image_identity(normalized_path)
                    
                    identity_idx = identity_to_index.get(identity)
                    
                    if identity_idx is not None:
                        # Store feature for this identity
                        if identity_has_features[identity_idx]:
                            # Already have features for this identity - check if they match
                            existing_feature = consolidated_features[identity_idx]
                            new_feature = features[i]
                            if not np.allclose(existing_feature, new_feature, atol=1e-5):
                                print(f"    Warning: Different features for identity {identity} (path: {normalized_path})")
                        else:
                            consolidated_features[identity_idx] = features[i]
                            identity_has_features[identity_idx] = True
                            mapped_count += 1
                            total_features_loaded += 1
                
                # Also store in path_to_feature for reference
                path_to_feature[normalized_path] = features[i]
            
            print(f"  Mapped {mapped_count:,} features to identities")
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal features loaded: {total_features_loaded:,}")
    print(f"Identities with features: {np.sum(identity_has_features):,} / {num_identities:,}")
    
    # Save consolidated features
    print(f"\nSaving consolidated features to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, consolidated_features)
    
    # Save paths (order matching the features array)
    print(f"Saving paths to {output_paths_file}...")
    with open(output_paths_file, 'w') as f:
        for identity in unique_identities:
            path = identity_to_path.get(identity, '')
            f.write(f"{path}\n")
    
    # Save metadata
    print(f"Saving metadata to {output_metadata_file}...")
    metadata = {
        'num_identities': num_identities,
        'feature_dim': feature_dim,
        'identities_with_features': int(np.sum(identity_has_features)),
        'identities_without_features': int(np.sum(~identity_has_features)),
        'datasets_processed': [ff['dataset'] for ff in feature_files],
    }
    with open(output_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique identities: {num_identities:,}")
    print(f"Identities with features: {np.sum(identity_has_features):,}")
    print(f"Identities without features: {np.sum(~identity_has_features):,}")
    print(f"Coverage: {np.sum(identity_has_features) / num_identities * 100:.2f}%")
    print(f"Output file: {output_file}")
    print(f"Output shape: {consolidated_features.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()

