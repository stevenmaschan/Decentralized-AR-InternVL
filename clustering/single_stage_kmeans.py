#!/usr/bin/env python3
"""
Single-stage spherical k-means clustering for CLIP features.

This script performs:
- Single-stage spherical k-means clustering using faiss-gpu
- No coarse clustering (just one stage)

Saves:
- Clustering assignments
- Centroids
- K-means model
- Metadata
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm

# Fix OpenMP threading conflicts - set before importing faiss
os.environ['OMP_NUM_THREADS'] = '1'

# Fix multiprocessing deadlocks - use spawn instead of fork
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

try:
    import faiss
except ImportError:
    print("Error: faiss-gpu is not installed. Please install it with: pip install faiss-gpu")
    sys.exit(1)


def normalize_features(features):
    """Normalize features to unit length for cosine distance."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # Avoid division by zero
    return features / norms


def faiss_spherical_kmeans(features, n_clusters, n_iter=100, nredo=3, device='cuda:0', verbose=True):
    """
    Perform spherical k-means using faiss-gpu.
    
    Uses faiss.Kmeans with spherical=True for spherical k-means clustering.
    
    Args:
        features: numpy array of shape (N, D)
        n_clusters: number of clusters
        n_iter: number of iterations
        nredo: number of redo runs - runs k-means multiple times and keeps best result (default: 3)
        device: device string (e.g., 'cuda:0', 'cpu'). If CUDA device, uses GPU for faiss
        verbose: whether to print progress
    
    Returns:
        centroids: numpy array of shape (n_clusters, D) - normalized centroids
        assignments: numpy array of shape (N,) - cluster assignments
        kmeans_model: faiss.Kmeans model object
    """
    N, D = features.shape
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Determine if we should use GPU based on device
    use_gpu = device.startswith('cuda')
    if use_gpu:
        # Extract GPU index from device string (e.g., 'cuda:0' -> 0)
        try:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
        except (ValueError, IndexError):
            gpu_id = 0
        
        if verbose:
            print(f"Attempting to use GPU {gpu_id} for faiss (device: {device})")
            print(f"Note: If CUBLAS errors occur, will automatically fall back to CPU")
        gpu_param = True
    else:
        if verbose:
            print(f"Using CPU for faiss (device: {device})")
        gpu_param = False
    
    if verbose:
        print(f"Running faiss spherical k-means with {n_clusters} clusters on {N} samples...")
    
    # Convert to float32 for faiss
    features = features.astype(np.float32)
    
    # Use faiss.Kmeans with max_points_per_centroid set to a very large value
    # This allows all points to be assigned to any centroid without limitation
    max_points_per_centroid = 2**31 - 1  # Very large value (max int32)
    
    if verbose:
        print(f"Using spherical k-means (spherical=True), max_points_per_centroid={max_points_per_centroid}, seed=42, nredo={nredo}, gpu={gpu_param}")
        print(f"Note: First run may take 10-20 minutes for JIT compilation on new GPUs. Please wait...")
    
    # Initialize and train k-means
    # Use spherical k-means (built-in faiss spherical option)
    # spherical=True will normalize features internally
    # Always use verbose=True to see progress and detect if it's stuck
    kmeans = faiss.Kmeans(
        D,
        n_clusters,
        niter=n_iter,
        nredo=nredo,  # Run k-means nredo times and keep best result
        gpu=gpu_param,  # Force GPU usage
        verbose=True,  # Always verbose to see progress
        seed=42,
        spherical=True,  # Use spherical k-means (cosine distance)
        max_points_per_centroid=max_points_per_centroid
    )
    
    # Train k-means on features (spherical=True will normalize internally)
    if verbose:
        if gpu_param:
            print(f"Training k-means on GPU (this may take 10-20 minutes on first run due to JIT compilation)...")
            print(f"Watch for 'Clustering X points...' message - if you see it, GPU is working!")
        else:
            print(f"Training k-means on CPU...")
    
    try:
        kmeans.train(features)
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "CUBLAS" in str(e):
            if verbose:
                print(f"\nCUBLAS error detected: {e}")
                print(f"This is likely due to a CUDA/CUBLAS version mismatch.")
                print(f"Falling back to CPU clustering...")
            # Retry with CPU
            kmeans = faiss.Kmeans(
                D,
                n_clusters,
                niter=n_iter,
                nredo=nredo,
                gpu=False,  # Use CPU
                verbose=True,
                seed=42,
                spherical=True,  # Use spherical k-means
                max_points_per_centroid=max_points_per_centroid
            )
            kmeans.train(features)
        else:
            raise
    
    # Get centroids (spherical=True already normalizes them)
    centroids = kmeans.centroids
    
    # Normalize centroids to ensure they're unit vectors (spherical should do this, but double-check)
    centroids = normalize_features(centroids)
    
    # Normalize features for assignment (spherical k-means requires normalized features for search)
    features_normalized = normalize_features(features)
    
    # Assign points to clusters using normalized features
    # Use the kmeans index for assignment
    distances, assignments = kmeans.index.search(features_normalized, 1)
    assignments = assignments.flatten()
    
    if verbose:
        print(f"Clustering complete. Centroids shape: {centroids.shape}")
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        print(f"Cluster size range: [{cluster_sizes.min()}, {cluster_sizes.max()}]")
        print(f"Cluster size mean: {cluster_sizes.mean():.2f}, std: {cluster_sizes.std():.2f}")
    
    return centroids, assignments, kmeans


def main():
    parser = argparse.ArgumentParser(
        description="Single-stage spherical k-means clustering for CLIP features"
    )
    parser.add_argument(
        '--features-file',
        type=str,
        required=True,
        help='Path to npy file containing features (shape: N, D)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        required=True,
        help='Number of clusters'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory to save clustering results'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=100,
        help='Number of k-means iterations (default: 100)'
    )
    parser.add_argument(
        '--nredo',
        type=int,
        default=3,
        help='Number of redo runs - runs k-means multiple times and keeps best result (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (cuda:0, cuda:1, cpu, etc.) (default: cuda:0)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for output files (default: clustering)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SINGLE-STAGE SPHERICAL K-MEANS CLUSTERING")
    print("=" * 80)
    
    # Load features
    print(f"\nLoading features from {args.features_file}...")
    features = np.load(args.features_file)
    print(f"Loaded features: shape {features.shape}")
    N, D = features.shape
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Run spherical k-means
    print(f"\nRunning spherical k-means with {args.n_clusters} clusters...")
    centroids, assignments, kmeans_model = faiss_spherical_kmeans(
        features,
        n_clusters=args.n_clusters,
        n_iter=args.n_iter,
        nredo=args.nredo,
        device=args.device,
        verbose=True
    )
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Save centroids
    centroids_file = output_dir / f"{args.prefix}_centroids.npy"
    np.save(centroids_file, centroids)
    print(f"Saved centroids to {centroids_file}")
    
    # Save assignments
    assignments_file = output_dir / f"{args.prefix}_assignments.npy"
    np.save(assignments_file, assignments)
    print(f"Saved assignments to {assignments_file}")
    
    # Save k-means model (faiss Kmeans is not picklable; save index to .faiss if available)
    model_file = output_dir / f"{args.prefix}_kmeans_model.pkl"
    model_file_str = None
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(kmeans_model, f)
        model_file_str = str(model_file)
        print(f"Saved k-means model to {model_file}")
    except (TypeError, pickle.PicklingError) as e:
        if hasattr(kmeans_model, 'index') and kmeans_model.index is not None:
            faiss_file = output_dir / f"{args.prefix}_kmeans_index.faiss"
            faiss.write_index(kmeans_model.index, str(faiss_file))
            model_file_str = str(faiss_file)
            print(f"Note: k-means object not picklable; saved faiss index to {faiss_file}")
        else:
            print(f"Note: k-means model not saved ({e}). Centroids and assignments are sufficient for inference.")
    
    # Save metadata
    metadata = {
        "n_clusters": args.n_clusters,
        "n_samples": int(N),
        "feature_dim": int(D),
        "n_iter": args.n_iter,
        "nredo": args.nredo,
        "device": args.device,
        "centroids_file": str(centroids_file),
        "assignments_file": str(assignments_file),
        "kmeans_model_file": model_file_str,
        "features_file": args.features_file,
    }
    
    metadata_file = output_dir / f"{args.prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Print cluster statistics
    print(f"\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)
    cluster_sizes = np.bincount(assignments, minlength=args.n_clusters)
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Total samples: {N:,}")
    print(f"Cluster size - min: {cluster_sizes.min():,}, max: {cluster_sizes.max():,}")
    print(f"Cluster size - mean: {cluster_sizes.mean():.2f}, std: {cluster_sizes.std():.2f}")
    print(f"Empty clusters: {np.sum(cluster_sizes == 0)}")
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()














