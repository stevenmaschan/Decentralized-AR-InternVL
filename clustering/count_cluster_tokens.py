#!/usr/bin/env python3
"""
Count entries and tokens for each cluster partition.

This script:
1. Loads InternVL tokenizer
2. Processes all JSONL files in cluster-0 and cluster-1 directories
3. Counts entries and tokens for each cluster
"""

import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict


def count_tokens_in_file(jsonl_file, tokenizer, show_progress=False):
    """Count entries and tokens in a JSONL file."""
    entries = 0
    tokens = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        # Count total lines for progress bar
        if show_progress:
            f.seek(0)
            total_lines = sum(1 for line in f if line.strip())
            f.seek(0)
            iter_lines = tqdm(f, desc=f"  {jsonl_file.stem}", total=total_lines, leave=False)
        else:
            iter_lines = f
        
        for line in iter_lines:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line.strip())
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
        '--data-dir',
        type=str,
        default='/media/data2/maschan/internvl/data',
        help='Root directory containing cluster-0 and cluster-1 directories'
    )
    
    args = parser.parse_args()
    
    # Initialize InternVL tokenizer
    print("Loading InternVL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        trust_remote_code=True,
        use_fast=False
    )
    
    data_dir = Path(args.data_dir)
    
    # Load dataset mixture to get repeat factors
    dataset_mixture_file = data_dir / 'dataset_mixture.json'
    dataset_mixture = {}
    if dataset_mixture_file.exists():
        with open(dataset_mixture_file, 'r') as f:
            dataset_mixture = json.load(f)
        print(f"Loaded dataset_mixture.json with {len(dataset_mixture)} datasets")
    else:
        print(f"Warning: dataset_mixture.json not found at {dataset_mixture_file}")
    
    # Dataset name mapping (for cases where file name differs from dataset name in mixture)
    dataset_name_mapping = {
        'infographicsvqa': 'infovqa',
        'refcoco+': 'refcoco_plus',
    }
    dataset_name_mapping_reverse = {v: k for k, v in dataset_name_mapping.items()}
    
    # Process each cluster
    cluster_stats = {}
    
    for cluster_id in [0, 1]:
        cluster_dir = data_dir / f"cluster-{cluster_id}"
        
        if not cluster_dir.exists():
            print(f"Warning: Cluster directory not found: {cluster_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing cluster-{cluster_id}")
        print(f"{'='*80}")
        
        # Find all JSONL files in cluster directory
        jsonl_files = sorted(cluster_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No JSONL files found in {cluster_dir}")
            continue
        
        print(f"Found {len(jsonl_files)} dataset files")
        
        cluster_entries = 0
        cluster_tokens = 0
        dataset_stats = []
        
        for jsonl_file in tqdm(jsonl_files, desc=f"Processing cluster-{cluster_id} files"):
            dataset_name = jsonl_file.stem
            entries, tokens = count_tokens_in_file(jsonl_file, tokenizer, show_progress=True)
            
            cluster_entries += entries
            cluster_tokens += tokens
            
            dataset_stats.append({
                'dataset': dataset_name,
                'entries': entries,
                'tokens': tokens
            })
            
            print(f"  {dataset_name:<25} {entries:>10,} entries, {tokens:>15,} tokens")
        
        cluster_stats[cluster_id] = {
            'total_entries': cluster_entries,
            'total_tokens': cluster_tokens,
            'dataset_stats': dataset_stats
        }
        
        print(f"\nCluster {cluster_id} totals:")
        print(f"  Entries: {cluster_entries:,}")
        print(f"  Tokens: {cluster_tokens:,}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Dataset':<25} {'Cluster 0':>20} {'Cluster 1':>20}")
    print(f"{'':25} {'Entries':>10} {'Tokens':>15} {'Entries':>10} {'Tokens':>15}")
    print("-" * 80)
    
    # Get all unique datasets
    all_datasets = set()
    for cluster_id in [0, 1]:
        if cluster_id in cluster_stats:
            for ds_stat in cluster_stats[cluster_id]['dataset_stats']:
                all_datasets.add(ds_stat['dataset'])
    
    for dataset in sorted(all_datasets):
        cluster0_entries = 0
        cluster0_tokens = 0
        cluster1_entries = 0
        cluster1_tokens = 0
        
        if 0 in cluster_stats:
            for ds_stat in cluster_stats[0]['dataset_stats']:
                if ds_stat['dataset'] == dataset:
                    cluster0_entries = ds_stat['entries']
                    cluster0_tokens = ds_stat['tokens']
                    break
        
        if 1 in cluster_stats:
            for ds_stat in cluster_stats[1]['dataset_stats']:
                if ds_stat['dataset'] == dataset:
                    cluster1_entries = ds_stat['entries']
                    cluster1_tokens = ds_stat['tokens']
                    break
        
        print(f"{dataset:<25} {cluster0_entries:>10,} {cluster0_tokens:>15,} {cluster1_entries:>10,} {cluster1_tokens:>15,}")
    
    print("-" * 80)
    
    total_entries_0 = cluster_stats[0]['total_entries'] if 0 in cluster_stats else 0
    total_tokens_0 = cluster_stats[0]['total_tokens'] if 0 in cluster_stats else 0
    total_entries_1 = cluster_stats[1]['total_entries'] if 1 in cluster_stats else 0
    total_tokens_1 = cluster_stats[1]['total_tokens'] if 1 in cluster_stats else 0
    
    print(f"{'TOTAL':<25} {total_entries_0:>10,} {total_tokens_0:>15,} {total_entries_1:>10,} {total_tokens_1:>15,}")
    print("="*80)
    
    # Summary with repeat factors
    if dataset_mixture:
        print("\n" + "="*80)
        print("SUMMARY WITH REPEAT FACTORS")
        print("="*80)
        
        print(f"\n{'Dataset':<25} {'Repeat':>7} {'Cluster 0':>20} {'Cluster 1':>20} {'Total Adjusted':>20}")
        print(f"{'':25} {'Factor':>7} {'Tokens':>15} {'×':>1} {'Tokens':>15} {'×':>1} {'Tokens':>15}")
        print("-" * 95)
        
        total_cluster0_raw = 0
        total_cluster1_raw = 0
        total_cluster0_adj = 0
        total_cluster1_adj = 0
        
        for dataset in sorted(all_datasets):
            # Get original dataset name (handle mapping)
            original_dataset_name = dataset_name_mapping_reverse.get(dataset, dataset)
            if original_dataset_name not in dataset_mixture:
                continue
            
            repeat_factor = dataset_mixture[original_dataset_name].get('repeat_time', 1)
            
            cluster0_entries = 0
            cluster0_tokens = 0
            cluster1_entries = 0
            cluster1_tokens = 0
            
            if 0 in cluster_stats:
                for ds_stat in cluster_stats[0]['dataset_stats']:
                    if ds_stat['dataset'] == dataset:
                        cluster0_entries = ds_stat['entries']
                        cluster0_tokens = ds_stat['tokens']
                        break
            
            if 1 in cluster_stats:
                for ds_stat in cluster_stats[1]['dataset_stats']:
                    if ds_stat['dataset'] == dataset:
                        cluster1_entries = ds_stat['entries']
                        cluster1_tokens = ds_stat['tokens']
                        break
            
            adjusted_0 = cluster0_tokens * repeat_factor
            adjusted_1 = cluster1_tokens * repeat_factor
            adjusted_total = adjusted_0 + adjusted_1
            
            total_cluster0_raw += cluster0_tokens
            total_cluster1_raw += cluster1_tokens
            total_cluster0_adj += adjusted_0
            total_cluster1_adj += adjusted_1
            
            if cluster0_tokens > 0 or cluster1_tokens > 0:
                print(f"{dataset:<25} {repeat_factor:>7} {cluster0_tokens:>15,} ×{repeat_factor:>2} {cluster1_tokens:>15,} ×{repeat_factor:>2} {adjusted_total:>15,}")
        
        print("-" * 95)
        print(f"{'TOTAL (RAW)':<25} {'':>7} {total_cluster0_raw:>15,} {'':>3} {total_cluster1_raw:>15,} {'':>3} {total_cluster0_raw + total_cluster1_raw:>15,}")
        print(f"{'TOTAL (ADJUSTED)':<25} {'':>7} {total_cluster0_adj:>15,} {'':>3} {total_cluster1_adj:>15,} {'':>3} {total_cluster0_adj + total_cluster1_adj:>15,}")
        print("="*95)
        
        print(f"\nCluster distribution (with repeat factors):")
        total_adjusted = total_cluster0_adj + total_cluster1_adj
        if total_adjusted > 0:
            pct_0 = (total_cluster0_adj / total_adjusted) * 100
            pct_1 = (total_cluster1_adj / total_adjusted) * 100
            print(f"  Cluster 0: {pct_0:.2f}% tokens ({total_cluster0_adj:,} tokens)")
            print(f"  Cluster 1: {pct_1:.2f}% tokens ({total_cluster1_adj:,} tokens)")
            print(f"  Grand Total: {total_adjusted:,} tokens")
    
    print(f"\nGrand totals (raw):")
    print(f"  Total entries: {total_entries_0 + total_entries_1:,}")
    print(f"  Total tokens: {total_tokens_0 + total_tokens_1:,}")
    print(f"\nCluster distribution (raw):")
    if total_entries_0 + total_entries_1 > 0:
        pct_0 = (total_entries_0 / (total_entries_0 + total_entries_1)) * 100
        pct_1 = (total_entries_1 / (total_entries_0 + total_entries_1)) * 100
        print(f"  Cluster 0: {pct_0:.2f}% entries, {(total_tokens_0 / (total_tokens_0 + total_tokens_1) * 100):.2f}% tokens")
        print(f"  Cluster 1: {pct_1:.2f}% entries, {(total_tokens_1 / (total_tokens_0 + total_tokens_1) * 100):.2f}% tokens")


if __name__ == '__main__':
    main()

