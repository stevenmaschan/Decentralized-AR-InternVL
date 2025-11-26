#!/usr/bin/env python3
"""
Partition JSONL files based on k-means clustering.

This script:
1. Loads npy arrays with CLIP features (same order as JSONL entries)
2. Loads k-means centroids (fine and coarse)
3. Assigns images to clusters using cosine similarity (normalized dot product)
4. Partitions JSONL files based on coarse cluster assignments

Usage:
    python partition_jsonl_by_clusters.py \
        --dataset kvqa \
        --clustering-dir clustering \
        --data-dir /media/data2/maschan/internvl/data \
        --clustering-results-dir clustering
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
    """Load existing clustering results (centroids and assignments)."""
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


def partition_jsonl_file(
    jsonl_file,
    image_to_coarse_cluster,
    data_dir,
    dataset_name,
    n_coarse_clusters
):
    """Partition a JSONL file based on coarse cluster assignments."""
    # Create output files for each coarse cluster in cluster-0, cluster-1, etc. directories
    output_files = {}
    for cluster_id in range(n_coarse_clusters):
        cluster_dir = data_dir / f"cluster-{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        output_file = cluster_dir / f"{dataset_name}.jsonl"
        # Write mode - each dataset gets its own file in each cluster directory
        output_files[cluster_id] = open(output_file, 'w')
    
    # Count entries per cluster
    cluster_counts = defaultdict(int)
    
    # Read and partition JSONL file
    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                image_path = entry.get('image', '')
                
                if image_path and image_path in image_to_coarse_cluster:
                    coarse_cluster = image_to_coarse_cluster[image_path]
                    output_files[coarse_cluster].write(line)
                    cluster_counts[coarse_cluster] += 1
                # If image not found in clustering (no CLIP features), skip it - don't assign to any cluster
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {jsonl_file}")
                continue
    
    # Close all output files
    for f in output_files.values():
        f.close()
    
    return cluster_counts


def main():
    parser = argparse.ArgumentParser(
        description="Partition JSONL files based on k-means clustering"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name to process (e.g., kvqa, docvqa)'
    )
    parser.add_argument(
        '--clustering-dir',
        type=str,
        default='clustering',
        help='Directory containing dataset npy feature files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/media/data2/maschan/internvl/data',
        help='Root directory containing dataset JSONL files'
    )
    parser.add_argument(
        '--clustering-results-dir',
        type=str,
        default='clustering',
        help='Directory containing clustering results (centroids, mappings)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for clustering result files (default: clustering)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"PARTITIONING JSONL FILES BY K-MEANS CLUSTERS: {args.dataset}")
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
    
    # Step 2: Load features from npy array
    print("\n" + "=" * 80)
    print(f"Step 2: Loading features for {args.dataset}")
    print("=" * 80)
    
    clustering_dir = Path(args.clustering_dir)
    
    # Map dataset names to actual directory/feature file names
    dataset_name_map = {
        'infographicsvqa': 'infovqa',
        'allava_instruct_sampled': 'allava',  # Use allava directory but sampled features
        'refcoco+': 'refcoco_plus',
    }
    actual_dataset_name = dataset_name_map.get(args.dataset, args.dataset)
    
    # Special case: textvqa uses textcaps features
    if args.dataset == 'textvqa':
        feature_dataset_name = 'textcaps'
    else:
        feature_dataset_name = actual_dataset_name
    
    dataset_dir = clustering_dir / feature_dataset_name
    
    # Try different possible filenames
    possible_names = []
    if args.dataset == 'allava_instruct_sampled':
        # Special case: use sampled features
        possible_names = [
            f"allava_sampled_clip_features_array.npy",
            f"allava_sampled_clip_features.npy",
        ]
    elif args.dataset == 'refcoco+':
        # Special case: directory is refcoco_plus
        possible_names = [
            f"refcoco_plus_clip_features_array.npy",
            f"refcoco_plus_clip_features.npy",
        ]
    elif args.dataset == 'textvqa':
        # Special case: use textcaps features
        possible_names = [
            f"textcaps_clip_features_array.npy",
            f"textcaps_clip_features.npy",
        ]
    else:
        possible_names = [
            f"{feature_dataset_name}_clip_features_array.npy",
            f"{feature_dataset_name}_clip_features.npy",
        ]
    
    features_file = None
    for name in possible_names:
        candidate = dataset_dir / name
        if candidate.exists():
            features_file = candidate
            break
    
    if features_file is None:
        raise FileNotFoundError(
            f"Features file not found for {args.dataset} in {dataset_dir}. "
            f"Tried: {possible_names}"
        )
    
    print(f"Loading features from {features_file}...")
    features = np.load(features_file)
    print(f"Loaded features: shape {features.shape}")
    
    # Step 3: Load JSONL file to get image paths
    print("\n" + "=" * 80)
    print(f"Step 3: Loading JSONL file and extracting image paths")
    print("=" * 80)
    
    data_dir = Path(args.data_dir)
    
    # Try to find JSONL file path from dataset_mixture.json first
    jsonl_file = None
    dataset_mixture_file = data_dir / 'dataset_mixture.json'
    if dataset_mixture_file.exists():
        with open(dataset_mixture_file, 'r') as f:
            dataset_mixture = json.load(f)
            if args.dataset in dataset_mixture:
                jsonl_path = dataset_mixture[args.dataset]['annotation']
                jsonl_file = Path(jsonl_path)
                if jsonl_file.exists():
                    print(f"Found JSONL file from dataset_mixture.json: {jsonl_file}")
    
    # Fallback to standard path
    if jsonl_file is None or not jsonl_file.exists():
        # Try dataset name mappings
        dataset_name_map = {
            'ai2d': 'ai2diagram',
            'infovqa': 'infographicsvqa',
        }
        actual_dataset_name = dataset_name_map.get(args.dataset, args.dataset)
        jsonl_file = data_dir / actual_dataset_name / 'train.jsonl'
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    # Load JSONL entries and extract image paths (in order)
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
                print(f"Warning: Skipping invalid JSON line")
                continue
    
    print(f"Loaded {len(jsonl_entries)} JSONL entries")
    
    # Check if number of features matches number of entries
    if len(features) != len(jsonl_entries):
        print(f"Note: {len(features)} features but {len(jsonl_entries)} JSONL entries")
        print(f"Only entries with CLIP features will be assigned to clusters ({len(features)} entries)")
        # Use only the first N entries that have features (assumes same order)
        n_with_features = len(features)
        jsonl_entries = jsonl_entries[:n_with_features]
        image_paths = image_paths[:n_with_features]
    
    # Step 4: Assign to clusters
    print("\n" + "=" * 80)
    print("Step 4: Assigning to clusters")
    print("=" * 80)
    
    print("Assigning to fine clusters using cosine similarity...")
    fine_assignments = assign_to_fine_clusters(features, fine_centroids)
    
    print("Mapping to coarse clusters...")
    coarse_assignments = assign_to_coarse_clusters(fine_assignments, fine_to_coarse)
    
    # Build image path to cluster mapping - only for entries with features
    image_to_coarse_cluster = {}
    for image_path, coarse_cluster in zip(image_paths, coarse_assignments):
        if image_path:
            image_to_coarse_cluster[image_path] = int(coarse_cluster)
    
    print(f"Mapped {len(image_to_coarse_cluster)} images to clusters")
    
    # Print cluster distribution
    cluster_counts = np.bincount(coarse_assignments, minlength=n_coarse_clusters)
    print(f"\nCluster distribution:")
    for cluster_id, count in enumerate(cluster_counts):
        print(f"  Cluster {cluster_id}: {count} images")
    
    # Step 5: Partition JSONL file
    print("\n" + "=" * 80)
    print("Step 5: Partitioning JSONL file")
    print("=" * 80)
    
    partition_counts = partition_jsonl_file(
        jsonl_file,
        image_to_coarse_cluster,
        data_dir,
        args.dataset,
        n_coarse_clusters
    )
    
    print(f"\nPartitioned entries:")
    for cluster_id, count in sorted(partition_counts.items()):
        print(f"  Cluster {cluster_id}: {count} entries")
    
    print(f"\nPartitioned files saved to: {data_dir}/cluster-0, cluster-1, etc.")
    print(f"Done!")


if __name__ == '__main__':
    main()
