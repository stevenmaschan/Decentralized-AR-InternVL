#!/usr/bin/env python3
"""
Create t-SNE visualization from CLIP features.

This script:
1. Loads CLIP features from .npz file
2. Applies t-SNE dimensionality reduction
3. Creates and saves a scatter plot visualization
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_features(npz_file, max_samples=None, random_state=42):
    """Load features from .npz file."""
    print(f"Loading features from {npz_file}...")
    data = np.load(npz_file)
    
    # Check if it's our format (with 'features' and 'image_paths' keys)
    if 'features' in data and 'image_paths' in data:
        features = data['features']
        image_paths = data['image_paths']
        # Convert numpy array of strings to list
        if isinstance(image_paths, np.ndarray):
            image_paths = image_paths.tolist()
    else:
        # Old format: dictionary with image paths as keys
        image_paths = list(data.keys())
        features_list = [data[key] for key in image_paths]
        # Stack into a single array
        features = np.vstack(features_list)
    
    original_count = len(features)
    # Sample if needed
    if max_samples and original_count > max_samples:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.choice(original_count, max_samples, replace=False)
        features = features[indices]
        image_paths = [image_paths[i] for i in indices]
        print(f"Sampled {max_samples} from {original_count} feature vectors")
    else:
        print(f"Loaded {original_count} feature vectors")
    
    print(f"Using {len(features)} feature vectors of dimension {features.shape[1]}")
    return features, image_paths


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


def plot_tsne_with_metadata(features_2d, image_paths, metadata_file, output_file, 
                            title="t-SNE Visualization of CLIP Features"):
    """Create t-SNE plot with optional coloring based on metadata."""
    print(f"\nCreating plot with metadata...")
    
    # Try to load metadata if available
    color_labels = None
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # You can add custom coloring logic here based on metadata
            # For now, just use a single color
            print("Metadata loaded (no custom coloring applied)")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=10, c='blue')
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
        description="Create t-SNE visualization from CLIP features"
    )
    parser.add_argument(
        "--features_file",
        type=str,
        required=True,
        help="Path to .npz file containing CLIP features"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output plot file (default: features_file with .png extension)"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Optional metadata JSON file"
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
        base_name = os.path.splitext(args.features_file)[0]
        args.output_file = f"{base_name}_tsne.png"
    
    # Set title
    if args.title is None:
        dataset_name = os.path.basename(os.path.dirname(args.features_file))
        args.title = f"t-SNE Visualization: {dataset_name}"
    
    # Load features
    features, image_paths = load_features(
        args.features_file,
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
    if args.metadata_file:
        plot_tsne_with_metadata(
            features_2d,
            image_paths,
            args.metadata_file,
            args.output_file,
            title=args.title
        )
    else:
        plot_tsne(features_2d, args.output_file, title=args.title)
    
    # Save t-SNE coordinates
    coords_file = args.output_file.replace('.png', '_coordinates.npy')
    np.save(coords_file, features_2d)
    print(f"t-SNE coordinates saved to {coords_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()



