#!/usr/bin/env python3
"""
Unified script to count entries and tokens for each cluster partition.
Supports any number of clusters and any CLIP model type.

Usage:
    python count_cluster_tokens_unified.py \
        --clusters-dir data/clusters-2_balanced_kmeans_vit_base-patch16 \
        --output-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \
        --clip-model "ViT base-patch16"
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers is not installed. Please install it with: pip install transformers")
    exit(1)


def count_tokens_in_file(jsonl_file, tokenizer, show_progress=False):
    """
    Count entries and tokens in a JSONL file.
    
    Returns:
        entries: number of entries
        tokens: total token count
    """
    if not jsonl_file.exists():
        return 0, 0
    
    entries = 0
    tokens = 0
    
    iterator = open(jsonl_file, 'r')
    if show_progress:
        # Count lines first for progress bar
        with open(jsonl_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        iterator = tqdm(open(jsonl_file, 'r'), total=total_lines, desc=f"  Counting {jsonl_file.name}", leave=False)
    
    for line in iterator:
        if not line.strip():
            continue
        
        try:
            entry = json.loads(line)
            entries += 1
            
            # Count tokens in conversations
            conversations = entry.get('conversations', [])
            for conv in conversations:
                text = conv.get('value', '')
                if text:
                    # Tokenize the text
                    tokenized = tokenizer.encode(text, add_special_tokens=False)
                    tokens += len(tokenized)
        except json.JSONDecodeError:
            continue
    
    return entries, tokens


def main():
    parser = argparse.ArgumentParser(
        description="Count entries and tokens for each cluster partition"
    )
    parser.add_argument(
        '--clusters-dir',
        type=str,
        required=True,
        help='Directory containing cluster-0, cluster-1, etc. directories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for statistics file (default: same as clusters-dir)'
    )
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT base-patch16',
        help='CLIP model name for documentation (e.g., "ViT base-patch16", "ViT large-patch14-336", "RN-50")'
    )
    parser.add_argument(
        '--clustering-type',
        type=str,
        default='balanced k-means',
        help='Clustering type for documentation (e.g., "balanced k-means", "two-stage k-means")'
    )
    parser.add_argument(
        '--workspace-root',
        type=str,
        default='/home/zling/maschan/InternVL',
        help='Workspace root directory'
    )
    
    args = parser.parse_args()
    
    # Initialize InternVL tokenizer
    print("Loading InternVL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        trust_remote_code=True,
        use_fast=False
    )
    print("Tokenizer loaded.")
    
    clusters_dir = Path(args.clusters_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = clusters_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect cluster directories automatically
    cluster_dirs = sorted([d for d in clusters_dir.iterdir() if d.is_dir() and d.name.startswith('cluster-')])
    cluster_ids = []
    for cluster_dir in cluster_dirs:
        try:
            cluster_id = int(cluster_dir.name.split('-')[1])
            cluster_ids.append(cluster_id)
        except (IndexError, ValueError):
            continue
    cluster_ids = sorted(cluster_ids)
    n_clusters = len(cluster_ids)
    print(f"Detected {n_clusters} clusters: {cluster_ids}")
    
    if n_clusters == 0:
        print("Error: No cluster directories found!")
        return
    
    # Load dataset mixture files for each cluster to get repeat factors
    cluster_mixtures = {}
    for cluster_id in cluster_ids:
        mixture_file = clusters_dir / f"cluster-{cluster_id}" / "dataset_mixture.json"
        if mixture_file.exists():
            with open(mixture_file, 'r') as f:
                cluster_mixtures[cluster_id] = json.load(f)
            print(f"Loaded dataset_mixture.json for cluster-{cluster_id}")
        else:
            print(f"Warning: dataset_mixture.json not found for cluster-{cluster_id}")
            cluster_mixtures[cluster_id] = {}
    
    # Also load original dataset_mixture.json for reference
    original_mixture_file = Path(args.workspace_root) / "data" / "dense" / "dataset_mixture.json"
    original_mixture = {}
    if original_mixture_file.exists():
        with open(original_mixture_file, 'r') as f:
            original_mixture = json.load(f)
        print(f"Loaded original dataset_mixture.json")
    
    # Process each cluster
    cluster_stats = {}
    all_datasets = set()
    
    for cluster_id in cluster_ids:
        cluster_dir = clusters_dir / f"cluster-{cluster_id}"
        if not cluster_dir.exists():
            print(f"Warning: Cluster directory not found: {cluster_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing cluster-{cluster_id}")
        print(f"{'='*80}")
        
        dataset_stats = []
        total_entries = 0
        total_tokens = 0
        
        # Get all JSONL files in cluster directory
        jsonl_files = sorted(cluster_dir.glob("*.jsonl"))
        
        for jsonl_file in tqdm(jsonl_files, desc=f"Processing cluster-{cluster_id} datasets"):
            dataset_name = jsonl_file.stem
            all_datasets.add(dataset_name)
            
            entries, tokens = count_tokens_in_file(jsonl_file, tokenizer, show_progress=True)
            
            dataset_stats.append({
                'dataset': dataset_name,
                'entries': entries,
                'tokens': tokens
            })
            
            total_entries += entries
            total_tokens += tokens
        
        cluster_stats[cluster_id] = {
            'dataset_stats': dataset_stats,
            'total_entries': total_entries,
            'total_tokens': total_tokens
        }
        
        print(f"\nCluster-{cluster_id} summary:")
        print(f"  Total entries: {total_entries:,}")
        print(f"  Total tokens: {total_tokens:,}")
    
    # Generate statistics report
    output_file = output_dir / "cluster_token_statistics.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CLUSTER TOKEN STATISTICS - SUMMARY WITH REPEAT FACTORS\n")
        f.write("="*100 + "\n")
        f.write("\n")
        f.write("="*100 + "\n")
        f.write("RAW STATISTICS (without repeat factors)\n")
        f.write("="*100 + "\n")
        f.write("\n")
        
        # Build header for all clusters
        header_line = f"{'Dataset':<40}"
        subheader_line = f"{'':40}"
        for cluster_id in cluster_ids:
            header_line += f"{'Cluster ' + str(cluster_id):>20}"
            subheader_line += f"{'Entries':>10} {'Tokens':>15}"
        f.write(header_line + "\n")
        f.write(subheader_line + "\n")
        f.write("-"*100 + "\n")
        
        # Calculate totals
        total_entries_by_cluster = {}
        total_tokens_by_cluster = {}
        for cluster_id in cluster_ids:
            total_entries_by_cluster[cluster_id] = cluster_stats.get(cluster_id, {}).get('total_entries', 0)
            total_tokens_by_cluster[cluster_id] = cluster_stats.get(cluster_id, {}).get('total_tokens', 0)
        
        # Dataset name mapping for display
        dataset_display_map = {
            'ai2d': 'ai2diagram',
            'infographicsvqa': 'infographicsvqa',
        }
        
        for dataset in sorted(all_datasets):
            display_name = dataset_display_map.get(dataset, dataset)
            
            # Get stats for this dataset from all clusters
            entries_by_cluster = {}
            tokens_by_cluster = {}
            has_data = False
            
            for cluster_id in cluster_ids:
                entries_by_cluster[cluster_id] = 0
                tokens_by_cluster[cluster_id] = 0
                
                if cluster_id in cluster_stats:
                    for ds_stat in cluster_stats[cluster_id]['dataset_stats']:
                        if ds_stat['dataset'] == dataset:
                            entries_by_cluster[cluster_id] = ds_stat['entries']
                            tokens_by_cluster[cluster_id] = ds_stat['tokens']
                            if ds_stat['entries'] > 0:
                                has_data = True
                            break
            
            if has_data:
                line = f"{display_name:<40}"
                for cluster_id in cluster_ids:
                    line += f"{entries_by_cluster[cluster_id]:>10,} {tokens_by_cluster[cluster_id]:>15,}"
                f.write(line + "\n")
        
        f.write("-"*100 + "\n")
        total_line = f"{'TOTAL':<40}"
        for cluster_id in cluster_ids:
            total_line += f"{total_entries_by_cluster[cluster_id]:>10,} {total_tokens_by_cluster[cluster_id]:>15,}"
        f.write(total_line + "\n")
        
        grand_total_tokens = sum(total_tokens_by_cluster.values())
        f.write(f"{'Grand Total':<40} {'':>10} {grand_total_tokens:>15,}\n")
        f.write("="*100 + "\n")
        f.write("\n")
        
        # Adjusted statistics with repeat factors
        f.write("="*100 + "\n")
        f.write("ADJUSTED STATISTICS (with repeat factors)\n")
        f.write("="*100 + "\n")
        f.write("\n")
        
        # Build header for adjusted stats - simplified format
        adj_header = f"{'Dataset':<30} {'Repeat':>7}"
        for cluster_id in cluster_ids:
            adj_header += f" {'Cluster ' + str(cluster_id):>20}"
        adj_header += f" {'Total Adjusted':>20}\n"
        f.write(adj_header)
        
        adj_subheader = f"{'':30} {'Factor':>7}"
        for cluster_id in cluster_ids:
            adj_subheader += f"{'Tokens ×':>15} {'':>5}"
        adj_subheader += f"{'Tokens':>20}\n"
        f.write(adj_subheader)
        f.write("-"*100 + "\n")
        
        total_raw_by_cluster = {cid: 0 for cid in cluster_ids}
        total_adj_by_cluster = {cid: 0 for cid in cluster_ids}
        
        for dataset in sorted(all_datasets):
            display_name = dataset_display_map.get(dataset, dataset)
            
            # Get repeat factor from original mixture or cluster mixture
            repeat_factor = 1
            if dataset in original_mixture:
                repeat_factor = original_mixture[dataset].get('repeat_time', 1)
            else:
                # Try to get from any cluster mixture
                for cluster_id in cluster_ids:
                    if cluster_id in cluster_mixtures and dataset in cluster_mixtures[cluster_id]:
                        repeat_factor = cluster_mixtures[cluster_id][dataset].get('repeat_time', 1)
                        break
            
            # Get tokens for this dataset from all clusters
            tokens_by_cluster = {}
            has_data = False
            
            for cluster_id in cluster_ids:
                tokens_by_cluster[cluster_id] = 0
                if cluster_id in cluster_stats:
                    for ds_stat in cluster_stats[cluster_id]['dataset_stats']:
                        if ds_stat['dataset'] == dataset:
                            tokens_by_cluster[cluster_id] = ds_stat['tokens']
                            if ds_stat['tokens'] > 0:
                                has_data = True
                            break
            
            if has_data:
                adjusted_by_cluster = {cid: tokens_by_cluster[cid] * repeat_factor for cid in cluster_ids}
                adjusted_total = sum(adjusted_by_cluster.values())
                
                for cluster_id in cluster_ids:
                    total_raw_by_cluster[cluster_id] += tokens_by_cluster[cluster_id]
                    total_adj_by_cluster[cluster_id] += adjusted_by_cluster[cluster_id]
                
                line = f"{display_name:<30} {repeat_factor:>7}"
                for cluster_id in cluster_ids:
                    line += f"{tokens_by_cluster[cluster_id]:>15,} ×{repeat_factor:<2} "
                line += f"{adjusted_total:>15,}"
                f.write(line + "\n")
        
        f.write("-"*100 + "\n")
        total_raw_line = f"{'TOTAL (RAW)':<30} {'':>7}"
        for cluster_id in cluster_ids:
            total_raw_line += f"{total_raw_by_cluster[cluster_id]:>15,} {'':>5}"
        total_raw_line += f"{sum(total_raw_by_cluster.values()):>15,}"
        f.write(total_raw_line + "\n")
        
        total_adj_line = f"{'TOTAL (ADJUSTED)':<30} {'':>7}"
        for cluster_id in cluster_ids:
            total_adj_line += f"{total_adj_by_cluster[cluster_id]:>15,} {'':>5}"
        total_adj_line += f"{sum(total_adj_by_cluster.values()):>15,}"
        f.write(total_adj_line + "\n")
        f.write("="*100 + "\n")
        f.write("\n")
        
        # Cluster distribution
        total_adjusted = sum(total_adj_by_cluster.values())
        if total_adjusted > 0:
            f.write(f"Cluster distribution (with repeat factors):\n")
            for cluster_id in cluster_ids:
                pct = (total_adj_by_cluster[cluster_id] / total_adjusted) * 100
                f.write(f"  Cluster {cluster_id}: {pct:.2f}% tokens ({total_adj_by_cluster[cluster_id]:,} tokens)\n")
            f.write(f"  Grand Total: {total_adjusted:,} tokens\n")
            f.write("\n")
        
        # Notes
        f.write("="*100 + "\n")
        f.write("NOTES\n")
        f.write("="*100 + "\n")
        f.write("- Repeat factors are from dataset_mixture.json\n")
        f.write("- All token counts use InternVL2-8B tokenizer\n")
        f.write(f"- Cluster partitioning is based on {args.clustering_type}:\n")
        f.write(f"    * {args.clustering_type}: {n_clusters} clusters using spherical (cosine) distance\n")
        f.write(f"- CLIP features: {args.clip_model}\n")
        f.write("- Only entries with CLIP features were partitioned\n")
        f.write("="*100 + "\n")
    
    print(f"\nStatistics saved to: {output_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()



