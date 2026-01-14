#!/usr/bin/env python3
"""
Partition JSONL files based on two-stage k-means clustering.

This script:
1. Loads clustering results from two-stage k-means (fine + coarse centroids)
2. Loads NPZ files with CLIP features from dataset clip_features directories
3. Assigns images to fine clusters, then maps to coarse clusters using cosine similarity
4. Partitions JSONL files from data/dense based on coarse cluster assignments

Usage:
    python partition_jsonl_by_clusters_two_stage.py \
        --clustering-results-dir clustering/kmeans_vit-base-patch-16_256-fine_2-coarse
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
    """Load existing clustering results (centroids and mappings)."""
    results_dir = Path(clustering_results_dir)
    
    # Load fine centroids
    fine_centroids_file = results_dir / f"{prefix}_fine_centroids.npy"
    if not fine_centroids_file.exists():
        raise FileNotFoundError(f"Fine centroids not found: {fine_centroids_file}")
    
    fine_centroids = np.load(fine_centroids_file)
    print(f"Loaded fine centroids: shape {fine_centroids.shape}")
    
    # Load coarse centroids
    coarse_centroids_file = results_dir / f"{prefix}_coarse_centroids.npy"
    if not coarse_centroids_file.exists():
        raise FileNotFoundError(f"Coarse centroids not found: {coarse_centroids_file}")
    
    coarse_centroids = np.load(coarse_centroids_file)
    print(f"Loaded coarse centroids: shape {coarse_centroids.shape}")
    
    # Load fine-to-coarse mapping
    fine_to_coarse_file = results_dir / f"{prefix}_fine_to_coarse_mapping.npy"
    if not fine_to_coarse_file.exists():
        raise FileNotFoundError(f"Fine-to-coarse mapping not found: {fine_to_coarse_file}")
    
    fine_to_coarse = np.load(fine_to_coarse_file)
    print(f"Loaded fine-to-coarse mapping: shape {fine_to_coarse.shape}")
    
    return fine_centroids, coarse_centroids, fine_to_coarse


def assign_to_fine_clusters(features, fine_centroids):
    """
    Assign features to fine clusters using cosine similarity (normalized dot product).
    
    Args:
        features: numpy array of shape (N, D) - CLIP features
        fine_centroids: numpy array of shape (n_fine_clusters, D) - fine centroids
    
    Returns:
        assignments: numpy array of shape (N,) - fine cluster assignments
    """
    # Normalize features and centroids
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    centroids_norm = fine_centroids / np.linalg.norm(fine_centroids, axis=1, keepdims=True)
    
    # Compute cosine similarity (dot product of normalized vectors)
    # Shape: (N, n_fine_clusters)
    similarities = np.dot(features_norm, centroids_norm.T)
    
    # Assign to cluster with highest similarity (argmax)
    assignments = np.argmax(similarities, axis=1)
    
    return assignments


def assign_to_coarse_clusters(fine_assignments, fine_to_coarse):
    """Map fine cluster assignments to coarse cluster assignments."""
    return fine_to_coarse[fine_assignments]


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
        # Try to find the features array
        keys = list(data.keys())
        if len(keys) == 1:
            features = data[keys[0]]
        else:
            raise ValueError(f"Could not find features in NPZ file: {npz_file}")
    
    # Get paths - try different key names
    paths = None
    for key in ['image_paths', 'paths', 'path']:
        if key in data:
            paths_data = data[key]
            if isinstance(paths_data, np.ndarray):
                # Convert numpy array to list
                paths = paths_data.tolist()
            elif isinstance(paths_data, (list, tuple)):
                paths = list(paths_data)
            else:
                paths = [paths_data]
            break
    
    if paths is None:
        # Fallback: use indices
        paths = [f"image_{i}" for i in range(len(features))]
    
    # Ensure paths is a list of strings
    paths = [str(p) for p in paths]
    
    return features, paths


def normalize_image_path(path):
    """Normalize image path for matching (handle different path formats)."""
    if not path:
        return ""
    # Remove leading/trailing whitespace
    path = path.strip()
    # Normalize separators
    path = path.replace('\\', '/')
    # Remove leading './' or 'data/'
    if path.startswith('./'):
        path = path[2:]
    if path.startswith('data/'):
        path = path[5:]
    return path


def partition_jsonl_file(
    jsonl_file,
    image_to_coarse_cluster,
    output_dir,
    dataset_name,
    n_coarse_clusters
):
    """Partition a JSONL file based on coarse cluster assignments."""
    # Create output files for each coarse cluster
    output_files = {}
    for cluster_id in range(n_coarse_clusters):
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
                if image_path in image_to_coarse_cluster:
                    cluster = image_to_coarse_cluster[image_path]
                elif normalized_path in image_to_coarse_cluster:
                    cluster = image_to_coarse_cluster[normalized_path]
                else:
                    # Try matching with basename
                    basename = os.path.basename(image_path) if image_path else ""
                    for path, c in image_to_coarse_cluster.items():
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


def get_dataset_dir_name(dataset_name):
    """Get the actual dataset directory name from dataset name."""
    dataset_dir_map = {
        'ai2d': 'ai2diagram',
        'infographicsvqa': 'infographicsvqa',
        'infovqa': 'infographicsvqa',
        'refcoco+': 'refcoco+',
        'allava_instruct_sampled': 'allava_instruct_sampled',
    }
    return dataset_dir_map.get(dataset_name, dataset_name)


def get_npz_file_path(dataset_name, data_root):
    """Get the NPZ file path for a dataset."""
    actual_dir = get_dataset_dir_name(dataset_name)
    
    # Map dataset names to NPZ file names
    npz_name_map = {
        'ai2d': 'ai2d_clip_base-patch16_features.npz',
        'infographicsvqa': 'infographicsvqa_clip_base-patch16_features.npz',
        'infovqa': 'infographicsvqa_clip_base-patch16_features.npz',
        'aokvqa': 'aokvqa_clip_base-patch16_features.npz',  # Use full version (not unique) to include all entries
        'textvqa': 'textvqa_clip_base-patch16_features.npz',
        'textcaps': 'textcaps_clip_base-patch16_features.npz',
    }
    
    if dataset_name in npz_name_map:
        npz_filename = npz_name_map[dataset_name]
    else:
        npz_filename = f"{dataset_name}_clip_base-patch16_features.npz"
    
    npz_file = data_root / actual_dir / 'clip_features' / npz_filename
    
    return npz_file


def main():
    parser = argparse.ArgumentParser(
        description="Partition JSONL files based on two-stage k-means clustering"
    )
    parser.add_argument(
        '--clustering-results-dir',
        type=str,
        required=True,
        help='Directory containing two-stage clustering results (fine_centroids.npy, coarse_centroids.npy, fine_to_coarse_mapping.npy)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clusters-2_vit_b_16_256_fine_2_coarse',
        help='Output directory for partitioned JSONL files'
    )
    parser.add_argument(
        '--dense-dir',
        type=str,
        default='data/dense',
        help='Directory containing original JSONL files'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for clustering result files (default: clustering)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Specific datasets to process (default: all in dense directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PARTITIONING JSONL FILES BY TWO-STAGE K-MEANS CLUSTERS")
    print("=" * 80)
    
    # Step 1: Load clustering results
    print("\n" + "=" * 80)
    print("Step 1: Loading clustering results")
    print("=" * 80)
    fine_centroids, coarse_centroids, fine_to_coarse = load_clustering_results(
        args.clustering_results_dir, args.prefix
    )
    n_coarse_clusters = len(coarse_centroids)
    print(f"Number of coarse clusters: {n_coarse_clusters}")
    
    # Step 2: Find all JSONL files to process
    dense_dir = Path(args.dense_dir)
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.datasets:
        jsonl_files = [(dense_dir / f"{ds}.jsonl", ds) for ds in args.datasets]
    else:
        # Find all JSONL files in dense directory
        jsonl_files = []
        for jsonl_file in dense_dir.glob("*.jsonl"):
            dataset_name = jsonl_file.stem
            # Skip backup files
            if dataset_name.endswith('.backup'):
                continue
            jsonl_files.append((jsonl_file, dataset_name))
    
    jsonl_files = sorted(jsonl_files, key=lambda x: x[1])
    print(f"\nFound {len(jsonl_files)} datasets to process")
    
    # Process each dataset
    total_stats = defaultdict(lambda: defaultdict(int))
    
    for jsonl_file, dataset_name in tqdm(jsonl_files, desc="Processing datasets", unit="dataset"):
        print("\n" + "=" * 80)
        print(f"Processing: {dataset_name}")
        print("=" * 80)
        
        # Step 2: Load features from NPZ file
        npz_file = get_npz_file_path(dataset_name, data_root)
        
        if not npz_file.exists():
            print(f"Warning: NPZ file not found: {npz_file}")
            print(f"Skipping {dataset_name}")
            continue
        
        print(f"Loading features from {npz_file}...")
        try:
            features, feature_paths = load_features_from_npz(npz_file)
            print(f"Loaded {len(features)} features")
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            continue
        
        # Step 3: Load JSONL file
        if not jsonl_file.exists():
            print(f"Warning: JSONL file not found: {jsonl_file}")
            continue
        
        print(f"Loading JSONL file: {jsonl_file}")
        jsonl_entries = []
        jsonl_image_paths = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    jsonl_entries.append(entry)
                    jsonl_image_paths.append(entry.get('image', ''))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(jsonl_entries)} JSONL entries")
        
        # Step 4: Match features to JSONL entries by image path
        print("Matching features to JSONL entries...")
        
        # Create mapping from normalized paths to feature indices
        path_to_feature_idx = {}
        for idx, path in enumerate(tqdm(feature_paths, desc=f"  Building path index for {dataset_name}", leave=False)):
            normalized = normalize_image_path(path)
            path_to_feature_idx[path] = idx
            if normalized != path:
                path_to_feature_idx[normalized] = idx
        
        # Match JSONL entries to features
        matched_features = []
        matched_indices = []
        
        for i, jsonl_path in enumerate(tqdm(jsonl_image_paths, desc=f"  Matching {dataset_name} entries", leave=False)):
            normalized_jsonl_path = normalize_image_path(jsonl_path)
            
            # Try to find matching feature
            feature_idx = None
            if jsonl_path in path_to_feature_idx:
                feature_idx = path_to_feature_idx[jsonl_path]
            elif normalized_jsonl_path in path_to_feature_idx:
                feature_idx = path_to_feature_idx[normalized_jsonl_path]
            else:
                # Try basename matching
                basename = os.path.basename(jsonl_path) if jsonl_path else ""
                for path, idx in path_to_feature_idx.items():
                    if os.path.basename(path) == basename:
                        feature_idx = idx
                        break
            
            if feature_idx is not None:
                matched_features.append(features[feature_idx])
                matched_indices.append(i)
        
        if len(matched_features) == 0:
            print(f"Warning: No features matched for {dataset_name}")
            continue
        
        matched_features = np.array(matched_features)
        print(f"Matched {len(matched_features)} features to JSONL entries")
        
        # Step 5: Assign to clusters
        print("Assigning to clusters...")
        with tqdm(total=2, desc=f"  Assigning {dataset_name} to clusters", leave=False) as pbar:
            fine_assignments = assign_to_fine_clusters(matched_features, fine_centroids)
            pbar.update(1)
            coarse_assignments = assign_to_coarse_clusters(fine_assignments, fine_to_coarse)
            pbar.update(1)
        
        # Build image path to cluster mapping
        image_to_coarse_cluster = {}
        for jsonl_idx, coarse_cluster in zip(matched_indices, coarse_assignments):
            image_path = jsonl_image_paths[jsonl_idx]
            if image_path:
                image_to_coarse_cluster[image_path] = int(coarse_cluster)
                # Also add normalized version
                normalized = normalize_image_path(image_path)
                if normalized != image_path:
                    image_to_coarse_cluster[normalized] = int(coarse_cluster)
        
        print(f"Mapped {len(image_to_coarse_cluster)} images to clusters")
        
        # Print cluster distribution
        cluster_counts = np.bincount(coarse_assignments, minlength=n_coarse_clusters)
        print(f"\nCluster distribution for {dataset_name}:")
        for cluster_id, count in enumerate(cluster_counts):
            print(f"  Cluster {cluster_id}: {count} images")
            total_stats[cluster_id][dataset_name] = count
        
        # Step 6: Partition JSONL file
        print(f"Partitioning JSONL file...")
        partition_counts = partition_jsonl_file(
            jsonl_file,
            image_to_coarse_cluster,
            output_dir,
            dataset_name,
            n_coarse_clusters
        )
        
        print(f"Partitioned entries for {dataset_name}:")
        for cluster_id, count in sorted(partition_counts.items()):
            print(f"  Cluster {cluster_id}: {count} entries")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for cluster_id in range(n_coarse_clusters):
        print(f"\nCluster {cluster_id} totals:")
        total = sum(total_stats[cluster_id].values())
        print(f"  Total: {total} entries")
        for dataset, count in sorted(total_stats[cluster_id].items()):
            print(f"    {dataset}: {count}")
    
    print(f"\nPartitioned files saved to: {output_dir}/cluster-0, cluster-1, etc.")
    print("Done!")


if __name__ == '__main__':
    main()

