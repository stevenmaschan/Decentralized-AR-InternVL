#!/usr/bin/env python3
"""
Generate t-SNE plot from existing clustering results.

This script loads saved clustering results and generates t-SNE visualization.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Prioritize cuML t-SNE (GPU/CUDA)
USE_GPU_TSNE = False
GPU_TSNE_LIB = None
TSNE_AVAILABLE = False

try:
    from cuml.manifold import TSNE as cuTSNE
    USE_GPU_TSNE = True
    GPU_TSNE_LIB = 'cuml'
    TSNE_AVAILABLE = True
    print("GPU-accelerated t-SNE (RAPIDS cuML) available for CUDA.")
except ImportError:
    try:
        import tsnecuda
        USE_GPU_TSNE = True
        GPU_TSNE_LIB = 'tsnecuda'
        TSNE_AVAILABLE = True
        print("GPU t-SNE (tsnecuda) available.")
    except ImportError:
        try:
            from openTSNE import TSNE
            USE_GPU_TSNE = False
            GPU_TSNE_LIB = None
            TSNE_AVAILABLE = True
            print("CPU t-SNE (openTSNE) available.")
        except ImportError:
            print("Error: No t-SNE library found.")
            print("Please install cuML: conda install -c rapidsai -c conda-forge -c nvidia cuml")
            sys.exit(1)


def normalize_features(features):
    """Normalize features to unit length for cosine distance."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return features / norms


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plot from existing clustering results"
    )
    parser.add_argument(
        '--clustering-dir',
        type=str,
        required=True,
        help='Directory containing clustering results'
    )
    parser.add_argument(
        '--features-file',
        type=str,
        required=True,
        help='Path to npy file containing original features'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='clustering',
        help='Prefix for clustering files (default: clustering)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of samples to use for t-SNE (default: 5000)'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='Perplexity parameter for t-SNE (default: 30)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    clustering_dir = Path(args.clustering_dir)
    features_file = Path(args.features_file)
    
    # Load features
    print(f"Loading features from {features_file}...")
    features = np.load(features_file)
    print(f"Loaded features: shape {features.shape}")
    
    # Load clustering results
    print(f"\nLoading clustering results from {clustering_dir}...")
    fine_centroids = np.load(clustering_dir / f"{args.prefix}_fine_centroids.npy")
    fine_assignments = np.load(clustering_dir / f"{args.prefix}_fine_assignments.npy")
    fine_to_coarse = np.load(clustering_dir / f"{args.prefix}_fine_to_coarse_mapping.npy")
    coarse_centroids = np.load(clustering_dir / f"{args.prefix}_coarse_centroids.npy")
    
    print(f"Fine centroids: {fine_centroids.shape}")
    print(f"Fine assignments: {fine_assignments.shape}")
    print(f"Coarse centroids: {coarse_centroids.shape}")
    
    # Sample points if needed
    N = len(features)
    n_samples = args.n_samples
    if n_samples and N > n_samples:
        np.random.seed(args.random_state)
        sample_indices = np.random.choice(N, n_samples, replace=False)
        sampled_features = features[sample_indices]
        sampled_fine_assignments = fine_assignments[sample_indices]
        print(f"Sampled {n_samples} points from {N} total points for t-SNE visualization")
    else:
        sampled_features = features
        sampled_fine_assignments = fine_assignments
        sample_indices = np.arange(N)
        print(f"Using all {N} points for t-SNE visualization")
    
    # Normalize features
    print("Normalizing features...")
    sampled_features_normalized = normalize_features(sampled_features)
    fine_centroids_normalized = normalize_features(fine_centroids)
    
    # Combine sampled features and fine centroids
    combined_features = np.vstack([sampled_features_normalized, fine_centroids_normalized])
    combined_features = normalize_features(combined_features)
    
    print(f"\nApplying t-SNE to {len(combined_features)} points (samples + fine centroids)...")
    
    # Run t-SNE
    if USE_GPU_TSNE and GPU_TSNE_LIB == 'cuml':
        print("Using GPU-accelerated t-SNE (RAPIDS cuML) on CUDA...")
        combined_features_gpu = combined_features.astype(np.float32)
        perplexity_adj = min(args.perplexity, len(combined_features) - 1)
        
        tsne_model = cuTSNE(
            n_components=2,
            perplexity=perplexity_adj,
            learning_rate=200.0,
            n_iter=1000,
            metric='euclidean',
            random_state=args.random_state,
            verbose=True
        )
        tsne_coords = tsne_model.fit_transform(combined_features_gpu)
        
        # Convert cupy array to numpy
        try:
            import cupy as cp
            if isinstance(tsne_coords, cp.ndarray):
                tsne_coords = cp.asnumpy(tsne_coords)
        except ImportError:
            if hasattr(tsne_coords, 'get'):
                tsne_coords = tsne_coords.get()
        tsne_coords = np.asarray(tsne_coords)
        
    elif USE_GPU_TSNE and GPU_TSNE_LIB == 'tsnecuda':
        print("Using GPU-accelerated t-SNE (tsnecuda)...")
        combined_features_gpu = combined_features.astype(np.float32)
        perplexity_adj = min(args.perplexity, len(combined_features) - 1)
        
        tsne_model = tsnecuda.TSNE(
            n_components=2,
            perplexity=perplexity_adj,
            learning_rate=200.0,
            n_iter=1000,
            metric='euclidean',
            random_seed=args.random_state,
            verbose=1,
            device=0
        )
        tsne_coords = tsne_model.fit_transform(combined_features_gpu)
        tsne_coords = np.array(tsne_coords)
        
    else:
        print("Using CPU t-SNE (openTSNE)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(args.perplexity, len(combined_features) - 1),
            random_state=args.random_state,
            n_iter=1000,
            metric="cosine",
            verbose=True
        )
        tsne_embedding = tsne.fit(combined_features)
        tsne_coords = np.array(tsne_embedding)
    
    # Check and scale coordinates if needed
    coord_range = tsne_coords.max() - tsne_coords.min()
    if coord_range > 1000:
        print(f"Large coordinate range detected ({coord_range:.2f}), scaling down...")
        coord_center = (tsne_coords.max() + tsne_coords.min()) / 2.0
        coord_scale = 50.0 / (coord_range / 2.0)
        tsne_coords = (tsne_coords - coord_center) * coord_scale
    
    # Split back into samples and centroids
    n_sampled = len(sampled_features)
    sample_coords = tsne_coords[:n_sampled]
    fine_centroid_coords = tsne_coords[n_sampled:]
    
    # Get coarse assignments for sampled points
    sampled_coarse_assignments = fine_to_coarse[sampled_fine_assignments]
    
    # Create plots
    print("\nCreating combined clustering visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Fine clustering plot
    unique_fine = np.unique(sampled_fine_assignments)
    n_fine_colors = len(unique_fine)
    try:
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')
    colors_fine = [cmap(i / max(n_fine_colors - 1, 1)) for i in range(n_fine_colors)]
    
    for i, fine_cluster in enumerate(unique_fine):
        mask = sampled_fine_assignments == fine_cluster
        color_idx = i % len(colors_fine)
        ax1.scatter(
            sample_coords[mask, 0],
            sample_coords[mask, 1],
            c=[colors_fine[color_idx]],
            alpha=0.6,
            s=3
        )
    
    ax1.set_title(f'Fine Clustering (t-SNE)\n{n_samples} samples, {len(fine_centroids)} fine clusters', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Coarse clustering plot with fine centroids as stars
    unique_coarse = np.unique(sampled_coarse_assignments)
    n_coarse_colors = len(unique_coarse)
    try:
        cmap_coarse = plt.colormaps['Set1']
    except (AttributeError, KeyError):
        cmap_coarse = plt.cm.get_cmap('Set1')
    colors_coarse = [cmap_coarse(i / max(n_coarse_colors - 1, 1)) for i in range(n_coarse_colors)]
    
    for i, coarse_cluster in enumerate(unique_coarse):
        mask = sampled_coarse_assignments == coarse_cluster
        ax2.scatter(
            sample_coords[mask, 0],
            sample_coords[mask, 1],
            c=[colors_coarse[i]],
            alpha=0.6,
            s=3,
            label=f'Cluster {coarse_cluster}'
        )
    
    # Plot fine centroids as stars
    for i, fine_centroid_coord in enumerate(fine_centroid_coords):
        coarse_cluster = fine_to_coarse[i]
        ax2.scatter(
            fine_centroid_coord[0],
            fine_centroid_coord[1],
            c=[colors_coarse[coarse_cluster]],
            marker='*',
            s=500,
            edgecolors='black',
            linewidths=1,
            alpha=0.8
        )
    
    ax2.set_title(f'Coarse Clustering (t-SNE)\n{n_samples} samples, {len(coarse_centroids)} coarse clusters', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = clustering_dir / f"{args.prefix}_tsne_combined_clustering.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved t-SNE plot to {output_file}")
    plt.close()
    
    print("\nt-SNE visualization complete!")


if __name__ == '__main__':
    main()














