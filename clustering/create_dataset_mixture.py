#!/usr/bin/env python3
"""
Create dataset_mixture.json files for cluster directories.

This script:
1. Loads the original dataset_mixture.json from data/dense
2. Counts entries in each cluster's JSONL files
3. Creates dataset_mixture.json files for each cluster with updated paths and lengths

Works with any number of clusters (auto-detects cluster-0, cluster-1, etc.).

Usage:
    python create_dataset_mixture.py --output-dir data/clusters-2_balanced_kmeans_vit_base-patch16
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def count_jsonl_entries(jsonl_file):
    """Count the number of entries in a JSONL file."""
    count = 0
    if not jsonl_file.exists():
        return 0
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def create_dataset_mixture_for_cluster(
    cluster_dir,
    original_mixture,
    cluster_id
):
    """Create dataset_mixture.json for a specific cluster."""
    cluster_mixture = {}
    
    # Get all JSONL files in the cluster directory
    jsonl_files = sorted(cluster_dir.glob("*.jsonl"))
    
    print(f"\nProcessing cluster-{cluster_id}...")
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for jsonl_file in tqdm(jsonl_files, desc=f"  Counting entries in cluster-{cluster_id}", leave=False):
        dataset_name = jsonl_file.stem
        
        # Count entries in this cluster's JSONL file
        length = count_jsonl_entries(jsonl_file)
        
        # Get original dataset config (if exists)
        if dataset_name in original_mixture:
            original_config = original_mixture[dataset_name].copy()
        else:
            # Default config if not in original
            original_config = {
                "root": "/home/zling/maschan/InternVL",
                "data_augment": False,
                "max_dynamic_patch": 1,
                "repeat_time": 1,
            }
        
        # Update annotation path and length
        cluster_mixture[dataset_name] = {
            "root": original_config["root"],
            "annotation": str(jsonl_file.absolute()),
            "data_augment": original_config.get("data_augment", False),
            "max_dynamic_patch": original_config.get("max_dynamic_patch", 1),
            "repeat_time": original_config.get("repeat_time", 1),
            "length": length
        }
    
    # Write dataset_mixture.json
    output_file = cluster_dir / "dataset_mixture.json"
    with open(output_file, 'w') as f:
        json.dump(cluster_mixture, f, indent=2)
    
    print(f"Created {output_file}")
    print(f"  Total datasets: {len(cluster_mixture)}")
    total_entries = sum(d["length"] for d in cluster_mixture.values())
    print(f"  Total entries: {total_entries:,}")
    
    return cluster_mixture


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset_mixture.json files for cluster directories"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory containing cluster-0, cluster-1, etc. directories'
    )
    parser.add_argument(
        '--dense-dir',
        type=str,
        default='data/dense',
        help='Directory containing original dataset_mixture.json'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CREATING DATASET MIXTURE JSON FILES FOR CLUSTERS")
    print("=" * 80)
    
    # Load original dataset_mixture.json
    dense_dir = Path(args.dense_dir)
    original_mixture_file = dense_dir / "dataset_mixture.json"
    
    if not original_mixture_file.exists():
        print(f"Warning: Original dataset_mixture.json not found at {original_mixture_file}")
        print("Using default configuration values")
        original_mixture = {}
    else:
        with open(original_mixture_file, 'r') as f:
            original_mixture = json.load(f)
        print(f"Loaded original dataset_mixture.json with {len(original_mixture)} datasets")
    
    # Process each cluster - auto-detect clusters
    output_dir = Path(args.output_dir)
    
    # Find all cluster directories
    cluster_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('cluster-')])
    
    if not cluster_dirs:
        print(f"Error: No cluster directories found in {output_dir}")
        return
    
    print(f"\nFound {len(cluster_dirs)} cluster directories")
    
    for cluster_dir in cluster_dirs:
        cluster_id = int(cluster_dir.name.split('-')[1])
        create_dataset_mixture_for_cluster(
            cluster_dir,
            original_mixture,
            cluster_id
        )
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()





