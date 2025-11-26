#!/usr/bin/env python3
"""
Create t-SNE visualization from a single feature array file.

This script:
1. Loads features from .npy array file
2. Samples N random features
3. Applies t-SNE dimensionality reduction
4. Creates and saves a scatter plot visualization
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_features(array_file, max_samples=None, random_state=42):
    """Load features from .npy array file."""
    print(f"Loading features from {array_file}...")
    features = np.load(array_file)
    
    original_count = len(features)
    print(f"Loaded {original_count:,} feature vectors of dimension {features.shape[1]}")
    
    # Sample if needed
    if max_samples and original_count > max_samples:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.choice(original_count, max_samples, replace=False)
        features = features[indices]
        print(f"Sampled {max_samples:,} from {original_count:,} feature vectors")
    else:
        print(f"Using all {original_count:,} feature vectors")
    
    print(f"Using {len(features):,} feature vectors of dimension {features.shape[1]}")
    return features


def apply_tsne(features, n_components=2, perplexity=30, random_state=42, max_iter=1000):
    """Apply t-SNE dimensionality reduction."""
    print(f"\nApplying t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        verbose=1
    )
    
    features_2d = tsne.fit_transform(features)
    print(f"t-SNE completed. Shape: {features_2d.shape}")
    
    return features_2d


def plot_tsne(features_2d, output_file, title="t-SNE Visualization of CLIP Features"):
    """Create and save t-SNE scatter plot."""
    print(f"\nCreating plot...")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=10)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create t-SNE visualization from feature array"
    )
    parser.add_argument(
        "--array_file",
        type=str,
        required=True,
        help="Path to .npy file containing feature array"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output plot file (default: array_file with _tsne.png extension)"
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for t-SNE (default: 1000)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: auto-generated)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for large datasets)"
    )
    
    args = parser.parse_args()
    
    # Set output file
    if args.output_file is None:
        base_name = os.path.splitext(args.array_file)[0]
        args.output_file = f"{base_name}_tsne.png"
    
    # Set title
    if args.title is None:
        dataset_name = os.path.basename(os.path.dirname(args.array_file))
        args.title = f"t-SNE Visualization: {dataset_name}"
    
    # Load features
    features = load_features(
        args.array_file,
        max_samples=args.max_samples,
        random_state=args.random_state
    )
    
    # Apply t-SNE
    features_2d = apply_tsne(
        features,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    
    # Create plot
    plot_tsne(features_2d, args.output_file, title=args.title)
    
    # Save t-SNE coordinates
    coords_file = args.output_file.replace('.png', '_coordinates.npy')
    np.save(coords_file, features_2d)
    print(f"t-SNE coordinates saved to {coords_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

