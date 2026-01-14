#!/usr/bin/env python3
"""
Partition JSONL files based on single-stage balanced k-means clustering.

This script:
1. Loads clustering results from single-stage balanced k-means (centroids only)
2. Loads NPZ files with CLIP features from dataset clip_features directories
3. Assigns images to clusters using cosine similarity
4. Partitions JSONL files from data/dense based on cluster assignments

Usage:
    python partition_jsonl_by_balanced_kmeans.py --clustering-results-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_clustering_results(clustering_results_dir, prefix="clustering"):
    """Load existing clustering results (centroids only for single-stage)."""
    results_dir = Path(clustering_results_dir)
    
    # Load centroids
    centroids_file = results_dir / f"{prefix}_centroids.npy"
    if not centroids_file.exists():
        raise FileNotFoundError(f"Centroids not found: {centroids_file}")
    
    centroids = np.load(centroids_file)
    print(f"Loaded centroids: shape {centroids.shape}")
    
    return centroids


def assign_to_clusters(features, centroids):
    """
    Assign features to clusters using cosine similarity (normalized dot product).
    
    Args:
        features: numpy array of shape (N, D) - CLIP features
        centroids: numpy array of shape (n_clusters, D) - cluster centroids
    
    Returns:
        assignments: numpy array of shape (N,) - cluster assignments
    """
    # Normalize features and centroids
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Compute cosine similarity (dot product of normalized vectors)
    # Shape: (N, n_clusters)
    similarities = np.dot(features_norm, centroids_norm.T)
    
    # Assign to cluster with highest similarity (argmax)
    assignments = np.argmax(similarities, axis=1)
    
    return assignments


def load_features_from_npz(npz_file):
    """
    Load features and paths from NPZ file.
    
    Returns:
        features: numpy array of shape (N, D)
        paths: list of image paths (strings)
    """
    data = np.load(npz_file, allow_pickle=True)
    
    # NPZ files contain 'features' and 'image_paths' or 'paths' keys
    if 'features' in data:
        features = data['features']
    elif 'arr_0' in data:
        features = data['arr_0']
    else:
        raise ValueError(f"Could not find features in NPZ file: {npz_file}")
    
    # Get image paths
    if 'image_paths' in data:
        paths = data['image_paths'].tolist()
    elif 'paths' in data:
        paths = data['paths'].tolist()
    elif 'image_path' in data:
        paths = data['image_path'].tolist()
    else:
        # If no paths stored, we'll need to extract from JSONL
        paths = None
    
    return features, paths


def get_npz_file_path(dataset_name, data_root, feature_type='base-patch16'):
    """Get the NPZ file path for a dataset."""
    # Map dataset names to directory names
    dataset_dir_map = {
        'ai2d': 'ai2diagram',
        'infographicsvqa': 'infographicsvqa',
        'infovqa': 'infographicsvqa',
        'refcoco+': 'refcoco+',
        'allava_instruct_sampled': 'allava_instruct_sampled',
    }
    
    actual_dir = dataset_dir_map.get(dataset_name, dataset_name)
    
    # Map feature types to file name suffixes
    feature_suffix_map = {
        'base-patch16': 'clip_base-patch16_features.npz',
        'rn50': 'clip_rn50_features.npz',
        'resnet50': 'clip_rn50_features.npz',
    }
    
    feature_suffix = feature_suffix_map.get(feature_type, f'clip_{feature_type}_features.npz')
    
    # Special handling for refcoco datasets (val, testA, testB all use same NPZ)
    if dataset_name.startswith('refcoco_'):
        # refcoco_val, refcoco_testA, refcoco_testB -> use refcoco/clip_features/refcoco_clip_base-patch16_features.npz
        actual_dir = 'refcoco'
        npz_filename = f'refcoco_{feature_suffix}'
    # Map dataset names to NPZ file names
    elif dataset_name in ['ai2d', 'infographicsvqa', 'infovqa', 'aokvqa', 'textvqa', 'textcaps']:
        npz_name_map = {
            'ai2d': f'ai2d_{feature_suffix}',
            'infographicsvqa': f'infographicsvqa_{feature_suffix}',
            'infovqa': f'infographicsvqa_{feature_suffix}',
            'aokvqa': f'aokvqa_{feature_suffix}',
            'textvqa': f'textvqa_{feature_suffix}',
            'textcaps': f'textcaps_{feature_suffix}',
        }
        npz_filename = npz_name_map[dataset_name]
    else:
        npz_filename = f"{dataset_name}_{feature_suffix}"
    
    npz_file = data_root / actual_dir / 'clip_features' / npz_filename
    
    return npz_file


def normalize_image_path(image_path):
    """Normalize image path for matching."""
    if not image_path:
        return ""
    # Remove leading/trailing whitespace
    image_path = image_path.strip()
    # Normalize path separators
    image_path = image_path.replace('\\', '/')
    return image_path


def partition_jsonl_file(
    jsonl_file,
    image_to_cluster,
    output_dir,
    dataset_name,
    n_clusters
):
    """Partition a JSONL file based on cluster assignments."""
    # Create output files for each cluster
    output_files = {}
    for cluster_id in range(n_clusters):
        cluster_dir = output_dir / f"cluster-{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        output_file = cluster_dir / f"{dataset_name}.jsonl"
        output_files[cluster_id] = open(output_file, 'w')
    
    # Count entries per cluster
    cluster_counts = defaultdict(int)
    skipped_count = 0
    
    # Read and partition JSONL file
    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                image_path = entry.get('image', '')
                
                # Normalize path for matching
                normalized_path = normalize_image_path(image_path)
                
                # Try to find cluster assignment
                cluster = None
                if image_path in image_to_cluster:
                    cluster = image_to_cluster[image_path]
                elif normalized_path in image_to_cluster:
                    cluster = image_to_cluster[normalized_path]
                else:
                    # Try matching with basename
                    basename = os.path.basename(image_path) if image_path else ""
                    for path, c in image_to_cluster.items():
                        if os.path.basename(path) == basename:
                            cluster = c
                            break
                
                if cluster is not None:
                    output_files[cluster].write(line)
                    cluster_counts[cluster] += 1
                else:
                    skipped_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {jsonl_file}")
                continue
    
    # Close all output files
    for f in output_files.values():
        f.close()
    
    if skipped_count > 0:
        print(f"  Warning: {skipped_count} entries skipped (no cluster assignment)")
    
    return cluster_counts


def main():
    parser = argparse.ArgumentParser(
        description="Partition JSONL files based on single-stage balanced k-means clustering"
    )
    parser.add_argument(
        '--clustering-results-dir',
        type=str,
        default='clustering/balanced_kmeans_unique_features_base-patch16_2_clusters',
        help='Directory containing clustering results (centroids.npy)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    parser.add_argument(
        '--dense-dir',
        type=str,
        default='data/dense',
        help='Directory containing dense JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clusters-2_balanced_kmeans_vit_base-patch16',
        help='Output directory for partitioned JSONL files'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for clustering result files (default: clustering)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Specific datasets to process (default: all found in dense directory)'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='base-patch16',
        help='Feature type to use: base-patch16, rn50, or resnet50 (default: base-patch16)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PARTITIONING JSONL FILES BY BALANCED K-MEANS CLUSTERS")
    print("=" * 80)
    
    # Step 1: Load clustering results
    print("\n" + "=" * 80)
    print("Step 1: Loading clustering results")
    print("=" * 80)
    centroids = load_clustering_results(args.clustering_results_dir, args.prefix)
    n_clusters = len(centroids)
    print(f"Number of clusters: {n_clusters}")
    
    # Step 2: Find all JSONL files to process
    dense_dir = Path(args.dense_dir)
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.datasets:
        jsonl_files = [(dense_dir / f"{dataset}.jsonl", dataset) for dataset in args.datasets]
    else:
        # Find all JSONL files in dense directory
        jsonl_files = [(f, f.stem) for f in sorted(dense_dir.glob("*.jsonl"))]
    
    print(f"\nFound {len(jsonl_files)} datasets to process")
    
    # Step 3: Process each dataset
    all_cluster_counts = defaultdict(lambda: defaultdict(int))
    
    for jsonl_file, dataset_name in tqdm(jsonl_files, desc="Processing datasets"):
        jsonl_file = Path(jsonl_file)
        if not jsonl_file.exists():
            print(f"Warning: JSONL file not found: {jsonl_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")
        
        # Load features from NPZ
        npz_file = get_npz_file_path(dataset_name, data_root, feature_type=args.feature_type)
        if not npz_file.exists():
            print(f"Warning: NPZ file not found: {npz_file}")
            continue
        
        print(f"Loading features from {npz_file}...")
        features, npz_paths = load_features_from_npz(npz_file)
        print(f"Loaded {len(features)} features")
        
        # Load JSONL entries
        print(f"Loading JSONL file: {jsonl_file}")
        jsonl_entries = []
        image_paths = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    jsonl_entries.append(entry)
                    image_paths.append(entry.get('image', ''))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(jsonl_entries)} JSONL entries")
        
        # Match features to JSONL entries
        print("Matching features to JSONL entries...")
        if npz_paths:
            # Build mapping from NPZ paths to features
            path_to_feature_idx = {}
            for idx, path in enumerate(npz_paths):
                if path:
                    normalized = normalize_image_path(path)
                    path_to_feature_idx[path] = idx
                    if normalized != path:
                        path_to_feature_idx[normalized] = idx
            
            # Match JSONL entries to features
            matched_features = []
            matched_paths = []
            matched_indices = []
            
            for idx, image_path in enumerate(tqdm(image_paths, desc=f"  Matching {dataset_name} entries", leave=False)):
                normalized_path = normalize_image_path(image_path)
                
                feature_idx = None
                if image_path in path_to_feature_idx:
                    feature_idx = path_to_feature_idx[image_path]
                elif normalized_path in path_to_feature_idx:
                    feature_idx = path_to_feature_idx[normalized_path]
                else:
                    # Try basename matching
                    basename = os.path.basename(image_path) if image_path else ""
                    for path, fidx in path_to_feature_idx.items():
                        if os.path.basename(path) == basename:
                            feature_idx = fidx
                            break
                
                if feature_idx is not None:
                    matched_features.append(features[feature_idx])
                    matched_paths.append(image_path)
                    matched_indices.append(idx)
            
            features = np.array(matched_features)
            image_paths = matched_paths
            jsonl_entries = [jsonl_entries[i] for i in matched_indices]
        else:
            # Assume same order - use first N features
            n_features = len(features)
            features = features[:len(jsonl_entries)]
            if len(features) < len(jsonl_entries):
                jsonl_entries = jsonl_entries[:len(features)]
                image_paths = image_paths[:len(features)]
        
        print(f"Matched {len(features)} features to JSONL entries")
        
        # Step 4: Assign to clusters
        print("Assigning to clusters...")
        assignments = assign_to_clusters(features, centroids)
        
        # Build image path to cluster mapping
        image_to_cluster = {}
        for image_path, cluster in zip(image_paths, assignments):
            if image_path:
                image_to_cluster[image_path] = int(cluster)
        
        print(f"Mapped {len(image_to_cluster)} images to clusters")
        
        # Print cluster distribution
        cluster_counts = np.bincount(assignments, minlength=n_clusters)
        print(f"\nCluster distribution for {dataset_name}:")
        for cluster_id, count in enumerate(cluster_counts):
            print(f"  Cluster {cluster_id}: {count} images")
            all_cluster_counts[cluster_id][dataset_name] += int(count)
        
        # Step 5: Partition JSONL file
        print("Partitioning JSONL file...")
        partition_counts = partition_jsonl_file(
            jsonl_file,
            image_to_cluster,
            output_dir,
            dataset_name,
            n_clusters
        )
        
        print(f"Partitioned entries for {dataset_name}:")
        for cluster_id in sorted(partition_counts.keys()):
            print(f"  Cluster {cluster_id}: {partition_counts[cluster_id]} entries")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for cluster_id in sorted(all_cluster_counts.keys()):
        total = sum(all_cluster_counts[cluster_id].values())
        print(f"\nCluster {cluster_id} totals:")
        print(f"  Total: {total} entries")
        for dataset, count in sorted(all_cluster_counts[cluster_id].items()):
            print(f"    {dataset}: {count}")
    
    print(f"\nPartitioned files saved to: {output_dir}/cluster-0, cluster-1, etc.")
    print("Done!")


if __name__ == '__main__':
    main()





