#!/usr/bin/env python3
"""
Convert existing paths.txt files to paths.json format for all datasets.
"""

import os
import json
from pathlib import Path


def convert_txt_to_json(dataset_dir, dataset_name):
    """Convert paths.txt to paths.json for a dataset."""
    txt_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features_array_paths.txt")
    json_file = os.path.join(dataset_dir, f"{dataset_name}_clip_features_array_paths.json")
    
    if not os.path.exists(txt_file):
        print(f"  ⚠️  {dataset_name}: txt file not found, skipping")
        return False
    
    if os.path.exists(json_file):
        print(f"  ⚠️  {dataset_name}: json file already exists, skipping")
        return False
    
    # Read paths from txt
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    # Write to json
    with open(json_file, 'w') as f:
        json.dump(paths, f, indent=2)
    
    print(f"  ✓ {dataset_name}: Converted {len(paths)} paths from txt to json")
    
    # Optionally remove txt file (uncomment if desired)
    # os.remove(txt_file)
    # print(f"    Removed txt file")
    
    return True


def main():
    clustering_dir = "/home/maschan/ddfm/InternVL/clustering"
    
    # Find all datasets with txt files
    datasets = []
    for item in os.listdir(clustering_dir):
        dataset_dir = os.path.join(clustering_dir, item)
        if os.path.isdir(dataset_dir):
            txt_file = os.path.join(dataset_dir, f"{item}_clip_features_array_paths.txt")
            if os.path.exists(txt_file):
                datasets.append(item)
    
    datasets.sort()
    
    print(f"Found {len(datasets)} datasets with txt files to convert:")
    for d in datasets:
        print(f"  - {d}")
    print()
    
    # Convert each dataset
    for dataset_name in datasets:
        dataset_dir = os.path.join(clustering_dir, dataset_name)
        convert_txt_to_json(dataset_dir, dataset_name)
    
    print(f"\n✓ Conversion complete!")


if __name__ == "__main__":
    main()

