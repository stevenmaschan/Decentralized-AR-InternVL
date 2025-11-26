#!/usr/bin/env python3
"""
Convert CLIP features from .npz format to a single numpy array (.npy).

This script:
1. Loads features from .npz file
2. Stacks them into a single numpy array
3. Saves as .npy file for faster loading
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def convert_npz_to_array(npz_file, output_file=None):
    """Convert .npz features to a single numpy array."""
    print(f"Loading features from {npz_file}...")
    data = np.load(npz_file)
    
    # Extract features and image paths
    image_paths = list(data.keys())
    print(f"Found {len(image_paths)} feature vectors")
    
    # Stack into a single array
    print("Stacking features into array...")
    features_list = []
    for key in tqdm(image_paths, desc="Loading features"):
        features_list.append(data[key])
    
    features = np.vstack(features_list)
    print(f"Stacked array shape: {features.shape}")
    
    # Set output file
    if output_file is None:
        base_name = os.path.splitext(npz_file)[0]
        output_file = f"{base_name}_array.npy"
    
    # Save as numpy array
    print(f"Saving to {output_file}...")
    np.save(output_file, features)
    print(f"Saved {features.shape[0]} features of dimension {features.shape[1]}")
    
    # Also save image paths for reference
    paths_file = output_file.replace('.npy', '_paths.txt')
    with open(paths_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    print(f"Saved image paths to {paths_file}")
    
    return features, image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert CLIP features from .npz to numpy array"
    )
    parser.add_argument(
        "--npz_file",
        type=str,
        required=True,
        help="Path to .npz file containing features"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .npy file (default: {npz_file}_array.npy)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        return
    
    convert_npz_to_array(args.npz_file, args.output_file)
    print("\nDone!")


if __name__ == "__main__":
    main()


