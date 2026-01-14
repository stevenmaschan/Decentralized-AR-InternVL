#!/usr/bin/env python3
"""
Two-stage k-means clustering for CLIP features.

This script performs:
1. Fine clustering: Uses faiss-gpu spherical k-means to cluster samples into fine centroids (default 1024)
2. Coarse clustering: Uses balanced k-means to cluster fine centroids into coarse clusters (default 2)

Saves:
- Fine clustering assignments
- Fine centroids
- Fine to coarse mapping
- Coarse centroids
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
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
                print("Error: No t-SNE implementation available.")
                print("Please install one of: tsnecuda, cuml, openTSNE, or scikit-learn")
                sys.exit(1)

try:
    import faiss
except ImportError:
    print("Error: faiss-gpu is not installed. Please install it with: pip install faiss-gpu")
    sys.exit(1)

# Add the balanced-kmeans path to sys.path
# Try to import, if it fails, try adding /tmp/balanced-kmeans to path
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
    # Note: faiss seed is set in the Kmeans constructor
    
    # Determine if we should use GPU based on device
    use_gpu = device.startswith('cuda') and faiss.get_num_gpus() > 0
    if use_gpu:
        # Extract GPU index from device string (e.g., 'cuda:0' -> 0)
        try:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
        except (ValueError, IndexError):
            gpu_id = 0
        if verbose:
            print(f"Using GPU {gpu_id} for faiss (device: {device})")
    else:
        if verbose:
            print(f"Using CPU for faiss (device: {device})")
    
    if verbose:
        print(f"Running faiss spherical k-means with {n_clusters} clusters on {N} samples...")
    
    # Convert to float32 for faiss
    features = features.astype(np.float32)
    
    # Use faiss.Kmeans with spherical=True and max_points_per_centroid set to a very large value
    # This allows all points to be assigned to any centroid without limitation
    max_points_per_centroid = 2**31 - 1  # Very large value (max int32)
    
    if verbose:
        print(f"Using spherical=True, max_points_per_centroid={max_points_per_centroid}, seed=42, nredo={nredo}")
    
    # Initialize and train k-means with spherical=True
    kmeans = faiss.Kmeans(
        D,
        n_clusters,
        niter=n_iter,
        nredo=nredo,  # Run k-means nredo times and keep best result
        gpu=use_gpu,
        verbose=verbose,
        seed=42,
        spherical=True,  # Use spherical k-means (cosine distance)
        max_points_per_centroid=max_points_per_centroid
    )
    
    # Train k-means
    kmeans.train(features)
    
    # Get centroids (already normalized when spherical=True)
    centroids = kmeans.centroids
    
    # Assign points to clusters
    # Use the kmeans index for assignment
    distances, assignments = kmeans.index.search(features, 1)
    assignments = assignments.flatten()
    
    if verbose:
        print(f"Fine clustering complete. Centroids shape: {centroids.shape}")
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        print(f"Cluster size range: [{cluster_sizes.min()}, {cluster_sizes.max()}]")
    
    return centroids, assignments, kmeans


def balanced_kmeans_clustering(fine_centroids, n_coarse_clusters, device='cuda:0', verbose=True):
    """
    Perform balanced k-means clustering on fine centroids.
    
    Args:
        fine_centroids: numpy array of shape (n_fine_clusters, D) - fine centroids
        n_coarse_clusters: number of coarse clusters
        device: device to use for balanced k-means
        verbose: whether to print progress
    
    Returns:
        coarse_centroids: torch tensor of shape (n_coarse_clusters, D)
        fine_to_coarse: numpy array of shape (n_fine_clusters,) - mapping from fine to coarse
        kmeans_model: KMeans model object from balanced-kmeans
    """
    if verbose:
        print(f"Running balanced k-means with {n_coarse_clusters} clusters on {len(fine_centroids)} fine centroids...")
    
    # Set random seeds for reproducibility (for balanced k-means initialization)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Convert to torch tensor
    fine_centroids_tensor = torch.from_numpy(fine_centroids).float()
    
    # Initialize balanced k-means
    kmeans = KMeans(
        n_clusters=n_coarse_clusters,
        device=torch.device(device),
        balanced=True
    )
    
    # Fit the model
    _ = kmeans.fit(
        X=fine_centroids_tensor,
        distance='cosine',  # Use cosine distance (spherical k-means)
        iter_limit=100,
        tqdm_flag=verbose
    )
    
    # Get coarse centroids
    coarse_centroids = kmeans.cluster_centers.cpu().numpy()
    
    # Predict assignments for fine centroids
    fine_to_coarse = kmeans.predict(
        X=fine_centroids_tensor,
        distance='cosine',
        balanced=True
    ).numpy()
    
    if verbose:
        print(f"Coarse clustering complete. Centroids shape: {coarse_centroids.shape}")
        print(f"Coarse cluster sizes: {np.bincount(fine_to_coarse)}")
    
    return coarse_centroids, fine_to_coarse, kmeans


def two_stage_kmeans(
    features,
    n_fine_clusters=1024,
    n_coarse_clusters=2,
    faiss_n_iter=100,
    faiss_nredo=3,
    device='cuda:0',
    verbose=True
):
    """
    Perform two-stage k-means clustering.
    
    Args:
        features: numpy array of shape (N, D) - CLIP features
        n_fine_clusters: number of fine clusters (default: 1024)
        n_coarse_clusters: number of coarse clusters (default: 2)
        faiss_n_iter: number of iterations for faiss k-means
        faiss_nredo: number of redo runs for faiss k-means (default: 3)
        device: device string for both faiss (GPU) and balanced k-means (default: 'cuda:0')
        verbose: whether to print progress
    
    Returns:
        fine_centroids: numpy array of shape (n_fine_clusters, D)
        fine_assignments: numpy array of shape (N,) - fine cluster assignments
        coarse_centroids: numpy array of shape (n_coarse_clusters, D)
        fine_to_coarse: numpy array of shape (n_fine_clusters,) - fine to coarse mapping
        fine_kmeans_model: faiss.Kmeans model object
        coarse_kmeans_model: KMeans model object from balanced-kmeans
    """
    # Set random seeds globally for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Stage 1: Fine clustering with faiss spherical k-means (uses GPU based on device)
    fine_centroids, fine_assignments, fine_kmeans_model = faiss_spherical_kmeans(
        features,
        n_clusters=n_fine_clusters,
        n_iter=faiss_n_iter,
        nredo=faiss_nredo,
        device=device,
        verbose=verbose
    )
    
    # Stage 2: Coarse clustering with balanced k-means
    coarse_centroids, fine_to_coarse, coarse_kmeans_model = balanced_kmeans_clustering(
        fine_centroids,
        n_coarse_clusters=n_coarse_clusters,
        device=device,
        verbose=verbose
    )
    
    return fine_centroids, fine_assignments, coarse_centroids, fine_to_coarse, fine_kmeans_model, coarse_kmeans_model


def save_results(
    output_dir,
    fine_centroids,
    fine_assignments,
    coarse_centroids,
    fine_to_coarse,
    fine_kmeans_model=None,
    coarse_kmeans_model=None,
    prefix="clustering"
):
    """
    Save clustering results to files.
    
    Args:
        output_dir: directory to save results
        fine_centroids: numpy array of fine centroids
        fine_assignments: numpy array of fine assignments
        coarse_centroids: numpy array of coarse centroids
        fine_to_coarse: numpy array of fine to coarse mapping
        fine_kmeans_model: faiss.Kmeans model object (optional)
        coarse_kmeans_model: KMeans model object from balanced-kmeans (optional)
        prefix: prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fine centroids
    fine_centroids_file = os.path.join(output_dir, f"{prefix}_fine_centroids.npy")
    np.save(fine_centroids_file, fine_centroids)
    print(f"Saved fine centroids to {fine_centroids_file}")
    
    # Save fine assignments
    fine_assignments_file = os.path.join(output_dir, f"{prefix}_fine_assignments.npy")
    np.save(fine_assignments_file, fine_assignments)
    print(f"Saved fine assignments to {fine_assignments_file}")
    
    # Save coarse centroids
    coarse_centroids_file = os.path.join(output_dir, f"{prefix}_coarse_centroids.npy")
    np.save(coarse_centroids_file, coarse_centroids)
    print(f"Saved coarse centroids to {coarse_centroids_file}")
    
    # Save fine to coarse mapping
    fine_to_coarse_file = os.path.join(output_dir, f"{prefix}_fine_to_coarse_mapping.npy")
    np.save(fine_to_coarse_file, fine_to_coarse)
    print(f"Saved fine to coarse mapping to {fine_to_coarse_file}")
    
    # Save metadata
    metadata = {
        'n_fine_clusters': len(fine_centroids),
        'n_coarse_clusters': len(coarse_centroids),
        'n_samples': len(fine_assignments),
        'feature_dim': fine_centroids.shape[1],
        'fine_centroids_file': fine_centroids_file,
        'fine_assignments_file': fine_assignments_file,
        'coarse_centroids_file': coarse_centroids_file,
        'fine_to_coarse_file': fine_to_coarse_file,
    }
    
    # Save full model objects if provided
    # Note: For faiss, we only save centroids as numpy (already done above)
    # For balanced k-means, we save the full model as pickle
    if coarse_kmeans_model is not None:
        coarse_model_file = os.path.join(output_dir, f"{prefix}_coarse_kmeans_model.pkl")
        coarse_kmeans_model.save(coarse_model_file)
        print(f"Saved coarse k-means model to {coarse_model_file}")
        metadata['coarse_kmeans_model_file'] = coarse_model_file
    
    import json
    metadata_file = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")


def plot_tsne_clustering(
    features,
    fine_assignments,
    fine_centroids,
    fine_to_coarse,
    coarse_centroids,
    output_dir,
    prefix="clustering",
    n_samples=5000,
    perplexity=30,
    random_state=42
):
    """
    Create t-SNE visualizations for fine and coarse clustering.
    
    Args:
        features: numpy array of shape (N, D) - all features
        fine_assignments: numpy array of shape (N,) - fine cluster assignments
        fine_centroids: numpy array of shape (n_fine_clusters, D) - fine centroids
        fine_to_coarse: numpy array of shape (n_fine_clusters,) - fine to coarse mapping
        coarse_centroids: numpy array of shape (n_coarse_clusters, D) - coarse centroids
        output_dir: directory to save plots
        prefix: prefix for output files
        n_samples: number of samples to use for t-SNE (default: 5000)
        perplexity: t-SNE perplexity parameter (default: 30)
        random_state: random seed for sampling and t-SNE (default: 42)
    """
    N = len(features)
    
    # Sample points if needed
    if n_samples and N > n_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(N, n_samples, replace=False)
        sampled_features = features[sample_indices]
        sampled_fine_assignments = fine_assignments[sample_indices]
        n_samples_used = n_samples
        print(f"Sampled {n_samples} points from {N} total points for t-SNE visualization")
    else:
        sampled_features = features
        sampled_fine_assignments = fine_assignments
        sample_indices = np.arange(N)
        n_samples_used = N
        print(f"Using all {N} points for t-SNE visualization")
    
    # Normalize features before t-SNE (to match clustering which uses normalized features)
    sampled_features_normalized = normalize_features(sampled_features)
    
    print(f"\nApplying t-SNE to {len(sampled_features_normalized)} points (samples only)...")
    print(f"Using {TSNE_LIBRARY} implementation")
    
    # Use appropriate t-SNE implementation based on what's available
    if TSNE_LIBRARY == "tsnecuda":
        # tsnecuda uses different parameter names
        tsne = TSNE_IMPL(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            learning_rate=200,
            n_iter=1000,
            verbose=1
        )
        tsne_coords = tsne.fit_transform(sampled_features_normalized.astype(np.float32))
    elif TSNE_LIBRARY == "cuml":
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
            
            # Use same parameters for both fine and coarse plots (single t-SNE computation)
            tsne = TSNE_IMPL(
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
            tsne_coords = cp.asnumpy(tsne_coords_gpu)
            
            # Check if results look reasonable (not all zeros or NaN)
            if np.any(np.isnan(tsne_coords)) or np.allclose(tsne_coords, 0):
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
            tsne_coords = np.array(tsne_embedding)
    elif TSNE_LIBRARY == "openTSNE":
        # openTSNE
        tsne = TSNE_IMPL(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            random_state=random_state,
            n_iter=1000,
            metric="cosine",
            verbose=True
        )
        tsne_embedding = tsne.fit(sampled_features_normalized)
        tsne_coords = np.array(tsne_embedding)
    else:
        # sklearn TSNE (doesn't support metric="cosine" directly, uses euclidean)
        tsne = TSNE_IMPL(
            n_components=2,
            perplexity=min(perplexity, len(sampled_features_normalized) - 1),
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )
        tsne_coords = tsne.fit_transform(sampled_features_normalized)
    
    # Use all coordinates as sample coordinates (no centroids)
    sample_coords = tsne_coords
    
    # Get coarse assignments for sampled points
    sampled_coarse_assignments = fine_to_coarse[fine_assignments[sample_indices]]
    
    # Create combined plot with both fine and coarse clustering
    print("\nCreating combined clustering visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get unique fine cluster labels and assign colors
    unique_fine = np.unique(sampled_fine_assignments)
    n_fine_colors = len(unique_fine)
    # Use compatible colormap access
    try:
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')
    colors_fine = [cmap(i / max(n_fine_colors - 1, 1)) for i in range(n_fine_colors)]
    
    # Plot 1: Fine clustering (no stars, no legend)
    for i, fine_cluster in enumerate(unique_fine):
        mask = sampled_fine_assignments == fine_cluster
        color_idx = i % len(colors_fine)
        ax1.scatter(
            sample_coords[mask, 0],
            sample_coords[mask, 1],
            c=[colors_fine[color_idx]],
            alpha=0.6,
            s=1
        )
    
    ax1.set_title(f'Fine Clustering (t-SNE)\n{n_samples_used} samples, {len(fine_centroids)} fine clusters', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coarse clustering with fine centroids as stars
    unique_coarse = np.unique(sampled_coarse_assignments)
    n_coarse_colors = len(unique_coarse)
    # Use compatible colormap access
    try:
        cmap_coarse = plt.colormaps['Set1']
    except (AttributeError, KeyError):
        cmap_coarse = plt.cm.get_cmap('Set1')
    colors_coarse = [cmap_coarse(i / max(n_coarse_colors - 1, 1)) for i in range(n_coarse_colors)]
    
    # Plot points colored by coarse cluster
    for i, coarse_cluster in enumerate(unique_coarse):
        mask = sampled_coarse_assignments == coarse_cluster
        color_idx = i % len(colors_coarse)
        ax2.scatter(
            sample_coords[mask, 0],
            sample_coords[mask, 1],
            c=[colors_coarse[color_idx]],
            label=f'Coarse Cluster {coarse_cluster}',
            alpha=0.6,
            s=1
        )
    
    ax2.set_title(f'Coarse Clustering (t-SNE)\n{n_samples_used} samples, {len(coarse_centroids)} coarse clusters', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    combined_plot_file = os.path.join(output_dir, f"{prefix}_tsne_combined_clustering.png")
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined clustering plot to {combined_plot_file}")
    plt.close()
    
    print("\nt-SNE visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage k-means clustering for CLIP features"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to numpy array file (.npy) containing CLIP features of shape (N, 768)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory name (will be created in clustering/ folder). If not specified, uses "clustering_results"'
    )
    parser.add_argument(
        '--n-fine-clusters',
        type=int,
        default=1024,
        help='Number of fine clusters (default: 1024)'
    )
    parser.add_argument(
        '--n-coarse-clusters',
        type=int,
        default=2,
        help='Number of coarse clusters (default: 2)'
    )
    parser.add_argument(
        '--faiss-n-iter',
        type=int,
        default=100,
        help='Number of iterations for faiss k-means (default: 100)'
    )
    parser.add_argument(
        '--faiss-nredo',
        type=int,
        default=1,
        help='Number of redo runs for faiss k-means - runs multiple times and keeps best result (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for both faiss (GPU) and balanced k-means (default: cuda:0). Use "cpu" for CPU-only.'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for output files (default: clustering)'
    )
    parser.add_argument(
        '--plot-tsne',
        action='store_true',
        help='Generate t-SNE visualizations of clustering results'
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
    
    # Set random seeds globally for full reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load features
    print(f"Loading features from {args.input_file}...")
    features = np.load(args.input_file)
    print(f"Loaded features with shape: {features.shape}")
    
    if len(features.shape) != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {features.shape}")
    
    if features.shape[1] != 768:
        print(f"Warning: Expected feature dimension 768, got {features.shape[1]}")
    
    # Set output directory - create a directory in clustering/ folder
    # Get the clustering directory (parent of this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clustering_dir = script_dir
    
    if args.output_dir is None:
        output_dir_name = "clustering_results"
    else:
        output_dir_name = args.output_dir
    
    # Create the full path in clustering/ folder
    args.output_dir = os.path.join(clustering_dir, output_dir_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Perform two-stage k-means
    fine_centroids, fine_assignments, coarse_centroids, fine_to_coarse, fine_kmeans_model, coarse_kmeans_model = two_stage_kmeans(
        features,
        n_fine_clusters=args.n_fine_clusters,
        n_coarse_clusters=args.n_coarse_clusters,
        faiss_n_iter=args.faiss_n_iter,
        faiss_nredo=args.faiss_nredo,
        device=args.device,
        verbose=True
    )
    
    # Save results
    save_results(
        args.output_dir,
        fine_centroids,
        fine_assignments,
        coarse_centroids,
        fine_to_coarse,
        fine_kmeans_model=fine_kmeans_model,
        coarse_kmeans_model=coarse_kmeans_model,
        prefix=args.prefix
    )
    
    # Generate t-SNE visualizations if requested
    if args.plot_tsne:
        plot_tsne_clustering(
            features,
            fine_assignments,
            fine_centroids,
            fine_to_coarse,
            coarse_centroids,
            args.output_dir,
            prefix=args.prefix,
            n_samples=args.n_tsne_samples,
            perplexity=args.tsne_perplexity,
            random_state=42
        )
    
    print("\nTwo-stage k-means clustering complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

