#!/usr/bin/env python3
"""
Single-stage spherical balanced k-means clustering for CLIP features.

This script performs balanced k-means clustering directly on CLIP features using cosine distance.
It uses the balanced-kmeans library which ensures clusters are roughly equal in size.

Saves:
- Cluster assignments
- Centroids
- KMeans model
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Try to import CUDA-accelerated t-SNE libraries in order of preference
TSNE_IMPL = None
TSNE_LIBRARY = None

# Try cuML first (more reliable installation)
try:
    from cuml.manifold import TSNE
    TSNE_IMPL = TSNE
    TSNE_LIBRARY = "cuml"
    print("Using RAPIDS cuML for GPU-accelerated t-SNE")
except (ImportError, OSError):
    # Try tsnecuda as fallback
    try:
        from tsnecuda import TSNE
        TSNE_IMPL = TSNE
        TSNE_LIBRARY = "tsnecuda"
        print("Using tsnecuda for GPU-accelerated t-SNE")
    except (ImportError, OSError):
        try:
            from openTSNE import TSNE
            TSNE_IMPL = TSNE
            TSNE_LIBRARY = "openTSNE"
            print("Warning: GPU t-SNE not available, using openTSNE (CPU) instead")
        except ImportError:
            try:
                from sklearn.manifold import TSNE
                TSNE_IMPL = TSNE
                TSNE_LIBRARY = "sklearn"
                print("Warning: GPU t-SNE not available, using sklearn.manifold.TSNE (CPU) instead")
            except ImportError:
                print("Warning: No t-SNE implementation available.")
                print("t-SNE plotting will be disabled.")
                TSNE_IMPL = None
                TSNE_LIBRARY = None

# Try to import balanced-kmeans
try:
    from kmeans_pytorch import KMeans
except ImportError:
    # Try adding the balanced-kmeans directory to path
    balanced_kmeans_path = '/tmp/balanced-kmeans'
    if os.path.exists(balanced_kmeans_path):
        sys.path.insert(0, balanced_kmeans_path)
        try:
            from kmeans_pytorch import KMeans
        except ImportError:
            print("Error: balanced-kmeans is not installed.")
            print("Please install it with: cd /tmp/balanced-kmeans && pip install --editable .")
            sys.exit(1)
    else:
        print("Error: balanced-kmeans is not installed.")
        print("Please install it from /tmp/balanced-kmeans")
        sys.exit(1)


def normalize_features(features):
    """Normalize features to unit length for cosine distance."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # Avoid division by zero
    return features / norms


def balanced_kmeans_clustering(features, n_clusters, device='cuda:0', iter_limit=100, verbose=True):
    """
    Perform balanced k-means clustering on features using cosine distance (spherical k-means).
    
    Args:
        features: numpy array of shape (N, D) - input features
        n_clusters: number of clusters
        device: device to use for balanced k-means (default: 'cuda:0')
        iter_limit: maximum number of iterations (default: 100)
        verbose: whether to print progress
    
    Returns:
        centroids: numpy array of shape (n_clusters, D) - cluster centroids
        assignments: numpy array of shape (N,) - cluster assignments
        kmeans_model: KMeans model object from balanced-kmeans
    """
    if verbose:
        print(f"Running balanced k-means with {n_clusters} clusters on {len(features)} samples...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Normalize features for cosine distance (spherical k-means)
    features_normalized = normalize_features(features)
    
    # Convert to torch tensor
    features_tensor = torch.from_numpy(features_normalized).float()
    
    # Initialize balanced k-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        device=torch.device(device),
        balanced=True
    )
    
    # Fit the model
    _ = kmeans.fit(
        X=features_tensor,
        distance='cosine',  # Use cosine distance (spherical k-means)
        iter_limit=iter_limit,
        tqdm_flag=verbose
    )
    
    # Get centroids
    centroids = kmeans.cluster_centers.cpu().numpy()
    
    # Predict assignments for all features
    assignments = kmeans.predict(
        X=features_tensor,
        distance='cosine',
        balanced=True
    ).numpy()
    
    if verbose:
        print(f"Clustering complete. Centroids shape: {centroids.shape}")
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
    
    return centroids, assignments, kmeans


def plot_tsne_clustering(
    features,
    assignments,
    centroids,
    output_dir,
    prefix="clustering",
    n_samples=5000,
    perplexity=30,
    random_state=42
):
    """
    Plot t-SNE visualization of clustering results.
    
    Args:
        features: numpy array of shape (N, D) - original features
        assignments: numpy array of shape (N,) - cluster assignments
        centroids: numpy array of shape (n_clusters, D) - cluster centroids
        output_dir: directory to save plots
        prefix: prefix for output files
        n_samples: number of samples to use for t-SNE (default: 5000)
        perplexity: perplexity parameter for t-SNE (default: 30)
        random_state: random state for reproducibility
    """
    if TSNE_IMPL is None:
        print("Warning: t-SNE not available, skipping visualization")
        return
    
    N = len(features)
    
    # Sample points if needed
    if n_samples and N > n_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(N, n_samples, replace=False)
        sampled_features = features[sample_indices]
        sampled_assignments = assignments[sample_indices]
        print(f"Sampled {n_samples} points from {N} total points for t-SNE visualization")
    else:
        sampled_features = features
        sampled_assignments = assignments
        sample_indices = np.arange(N)
        print(f"Using all {N} points for t-SNE visualization")
    
    # Normalize features for t-SNE
    sampled_features_normalized = normalize_features(sampled_features)
    
    # Run t-SNE
    print(f"Running t-SNE ({TSNE_LIBRARY}) with perplexity={perplexity}...")
    if TSNE_LIBRARY == "cuml":
        # RAPIDS cuML TSNE
        print("Warning: cuML TSNE may not work well with cosine distance.")
        print("Attempting cuML TSNE with adjusted parameters...")
        
        import cupy as cp
        try:
            # Convert to cupy array
            sampled_features_gpu = cp.asarray(sampled_features_normalized.astype(np.float32))
            
            # Adjust perplexity to be safe (cuML recommends < n_samples/3)
            max_perplexity = min((len(sampled_features_normalized) - 1) // 3, 50)
            safe_perplexity = min(perplexity, max_perplexity)
            
            if safe_perplexity < 5:
                safe_perplexity = 5  # Minimum perplexity
            
            print(f"Using perplexity={safe_perplexity} (adjusted from {perplexity})")
            
            # Use same parameters as two-stage script
            tsne = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                learning_rate=200,
                n_iter=1000,
                random_state=random_state,
                verbose=True,
                method='barnes_hut'
            )
            tsne_coords_gpu = tsne.fit_transform(sampled_features_gpu)
            # Convert back to numpy
            sample_coords = cp.asnumpy(tsne_coords_gpu)
            
            # Check if results look reasonable (not all zeros or NaN)
            if np.any(np.isnan(sample_coords)) or np.allclose(sample_coords, 0):
                raise ValueError("cuML TSNE produced invalid results (all zeros or NaN)")
                
        except Exception as e:
            print(f"Warning: cuML TSNE failed or produced poor results: {e}")
            print("Falling back to openTSNE (better for cosine distance)...")
            # Fallback to openTSNE
            from openTSNE import TSNE as openTSNE_impl
            tsne = openTSNE_impl(
                n_components=2,
                perplexity=min(perplexity, len(sampled_features_normalized) - 1),
                random_state=random_state,
                n_iter=1000,
                metric="cosine",
                verbose=True
            )
            tsne_embedding = tsne.fit(sampled_features_normalized)
            sample_coords = np.array(tsne_embedding)
    elif TSNE_LIBRARY == "tsnecuda":
        # tsnecuda
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            learning_rate=200,
            n_iter=1000,
            verbose=1
        )
        sample_coords = tsne.fit_transform(sampled_features_normalized.astype(np.float32))
    elif TSNE_LIBRARY == "openTSNE":
        # openTSNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            random_state=random_state,
            n_iter=1000,
            metric="cosine",
            verbose=True
        )
        tsne_embedding = tsne.fit(sampled_features_normalized)
        sample_coords = np.array(tsne_embedding)
    else:
        # sklearn TSNE (doesn't support metric="cosine" directly, uses euclidean)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )
        sample_coords = tsne.fit_transform(sampled_features_normalized)
    
    # Create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Generate colors for clusters
    n_clusters = len(centroids)
    try:
        # For matplotlib >= 3.7
        colors = plt.colormaps['tab20'](np.linspace(0, 1, n_clusters))
    except AttributeError:
        # Fallback for older matplotlib
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_clusters))
    
    # Plot points colored by cluster
    unique_clusters = np.unique(sampled_assignments)
    for i, cluster_id in enumerate(unique_clusters):
        mask = sampled_assignments == cluster_id
        color_idx = int(cluster_id) % len(colors)
        ax.scatter(
            sample_coords[mask, 0],
            sample_coords[mask, 1],
            c=[colors[color_idx]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=1
        )
    
    n_samples_used = len(sampled_assignments)
    ax.set_title(f'Balanced K-Means Clustering (t-SNE)\n{n_samples_used} samples, {n_clusters} clusters', 
                  fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{prefix}_tsne_clustering.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {plot_file}")
    plt.close()
    
    print("\nt-SNE visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Single-stage spherical balanced k-means clustering for CLIP features"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to numpy array file (.npy) containing CLIP features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory name (will be created in clustering/ folder). If not specified, uses "balanced_kmeans_results"'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=2,
        help='Number of clusters (default: 2)'
    )
    parser.add_argument(
        '--iter-limit',
        type=int,
        default=100,
        help='Maximum number of iterations for balanced k-means (default: 100)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    parser.add_argument(
        '--plot-tsne',
        action='store_true',
        help='Generate t-SNE visualization plots'
    )
    parser.add_argument(
        '--n-tsne-samples',
        type=int,
        default=5000,
        help='Number of samples to use for t-SNE visualization (default: 5000)'
    )
    parser.add_argument(
        '--tsne-perplexity',
        type=int,
        default=30,
        help='Perplexity parameter for t-SNE (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features from {args.input_file}...")
    features = np.load(args.input_file)
    print(f"Loaded features: shape {features.shape}, dtype {features.dtype}")
    
    # Determine output directory
    if args.output_dir is None:
        # Generate output directory name from input file
        input_name = Path(args.input_file).stem
        args.output_dir = f"balanced_kmeans_{input_name}_{args.n_clusters}_clusters"
    
    output_dir = Path("clustering") / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Run balanced k-means clustering
    centroids, assignments, kmeans_model = balanced_kmeans_clustering(
        features,
        n_clusters=args.n_clusters,
        device=args.device,
        iter_limit=args.iter_limit,
        verbose=True
    )
    
    # Save results
    prefix = "clustering"
    
    # Save centroids
    centroids_file = output_dir / f"{prefix}_centroids.npy"
    np.save(centroids_file, centroids)
    print(f"Saved centroids to {centroids_file}")
    
    # Save assignments
    assignments_file = output_dir / f"{prefix}_assignments.npy"
    np.save(assignments_file, assignments)
    print(f"Saved assignments to {assignments_file}")
    
    # Save kmeans model
    model_file = output_dir / f"{prefix}_kmeans_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(kmeans_model, f)
    print(f"Saved kmeans model to {model_file}")
    
    # Save metadata
    metadata = {
        'n_clusters': args.n_clusters,
        'n_samples': len(features),
        'feature_dim': features.shape[1],
        'device': args.device,
        'iter_limit': args.iter_limit,
        'cluster_sizes': np.bincount(assignments, minlength=args.n_clusters).tolist(),
        'clustering_method': 'single-stage spherical balanced k-means',
        'distance_metric': 'cosine',
    }
    
    metadata_file = output_dir / f"{prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Generate t-SNE visualization if requested
    if args.plot_tsne:
        plot_tsne_clustering(
            features,
            assignments,
            centroids,
            output_dir,
            prefix=prefix,
            n_samples=args.n_tsne_samples,
            perplexity=args.tsne_perplexity,
            random_state=42
        )
    
    print(f"\nClustering complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

