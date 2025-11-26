#!/usr/bin/env python3
"""
Create combined t-SNE visualization from multiple CLIP feature files.

This script:
1. Loads CLIP features from multiple .npz files
2. Combines them with dataset labels
3. Applies t-SNE dimensionality reduction
4. Creates a scatter plot with different colors for each dataset
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_features_from_multiple_files(feature_files, max_samples_per_dataset=None, random_state=42):
    """Load features from multiple .npz files and return combined features with labels."""
    all_features = []
    all_labels = []
    dataset_names = []
    
    if random_state is not None:
        np.random.seed(random_state)
    
    for dataset_name, feature_file in feature_files.items():
        print(f"Loading features from {feature_file}...")
        
        # Handle both .npz and .npy files
        if feature_file.endswith('.npy'):
            # Single array file
            features = np.load(feature_file)
        else:
            # .npz file (dictionary format)
            data = np.load(feature_file)
            
            # Check if it's our format (with 'features' and 'image_paths' keys)
            if 'features' in data and 'image_paths' in data:
                features = data['features']
            else:
                # Old format: dictionary with image paths as keys
                image_paths = list(data.keys())
                features_list = [data[key] for key in image_paths]
                features = np.vstack(features_list)
        
        # Sample if needed
        original_count = len(features)
        if max_samples_per_dataset and original_count > max_samples_per_dataset:
            indices = np.random.choice(original_count, max_samples_per_dataset, replace=False)
            features = features[indices]
            print(f"  Sampled {max_samples_per_dataset} from {original_count} feature vectors in {dataset_name}")
        else:
            print(f"  Loaded {original_count} feature vectors from {dataset_name}")
        
        all_features.append(features)
        all_labels.extend([dataset_name] * len(features))
        dataset_names.append(dataset_name)
    
    # Combine all features
    combined_features = np.vstack(all_features)
    print(f"\nTotal features: {len(combined_features)} of dimension {combined_features.shape[1]}")
    print(f"Dataset distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    
    return combined_features, np.array(all_labels), dataset_names


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


def plot_tsne_combined(features_2d, labels, dataset_names, output_file, title="Combined t-SNE Visualization"):
    """Create and save combined t-SNE scatter plot with different colors for each dataset."""
    print(f"\nCreating combined plot...")
    
    # Define colors for each dataset
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(dataset_names)}
    
    plt.figure(figsize=(14, 12))
    
    # Plot each dataset with different color
    for dataset_name in dataset_names:
        mask = labels == dataset_name
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=color_map[dataset_name],
            label=dataset_name,
            alpha=0.6,
            s=10
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create combined t-SNE visualization from multiple CLIP feature files"
    )
    parser.add_argument(
        "--feature_files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to .npz feature files (format: dataset_name:path/to/file.npz)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clustering/combined_tsne.png",
        help="Output plot file"
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
        default="Combined t-SNE Visualization of CLIP Features",
        help="Plot title"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per dataset (for large datasets, e.g., 10000)"
    )
    parser.add_argument(
        "--sample_datasets",
        type=str,
        nargs='+',
        default=None,
        help="List of dataset names to sample (if not provided, samples all datasets exceeding max_samples)"
    )
    
    args = parser.parse_args()
    
    # Parse feature files (format: dataset_name:path)
    feature_files = {}
    for item in args.feature_files:
        if ':' in item:
            dataset_name, file_path = item.split(':', 1)
            feature_files[dataset_name] = file_path
        else:
            # If no colon, use filename as dataset name
            dataset_name = os.path.splitext(os.path.basename(item))[0].replace('_clip_features', '')
            feature_files[dataset_name] = item
    
    print(f"Processing {len(feature_files)} datasets:")
    for name, path in feature_files.items():
        print(f"  {name}: {path}")
    
    # Determine which datasets to sample
    sample_datasets = args.sample_datasets if args.sample_datasets else None
    
    # Load features with optional sampling
    features, labels, dataset_names = load_features_from_multiple_files(
        feature_files,
        max_samples_per_dataset=args.max_samples,
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
    plot_tsne_combined(features_2d, labels, dataset_names, args.output_file, title=args.title)
    
    # Save t-SNE coordinates and labels
    coords_file = args.output_file.replace('.png', '_coordinates.npy')
    labels_file = args.output_file.replace('.png', '_labels.npy')
    np.save(coords_file, features_2d)
    np.save(labels_file, labels)
    print(f"t-SNE coordinates saved to {coords_file}")
    print(f"Labels saved to {labels_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()



