#!/usr/bin/env python3
"""
Sample random examples from ALLaVA instruct dataset until token count exceeds target.

This script:
1. Loads ALLaVA instruct training data
2. Randomly samples entries
3. Counts tokens for each entry
4. Keeps sampling until reaching target token count (~14M tokens)
5. Saves the sampled data to a new JSONL file
"""

import json
import os
import random
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


def load_allava_instruct(file_path):
    """Load all entries from ALLaVA instruct file."""
    print(f"Loading ALLaVA instruct from {file_path}...")
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(entries):,} entries")
    return entries


def count_entry_tokens(entry, tokenizer):
    """Count tokens in an entry's conversations."""
    conversations = entry.get('conversations', [])
    total_tokens = 0
    for conv in conversations:
        text = conv.get('value', '')
        if text:
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokenized)
    return total_tokens


def sample_until_target_tokens(entries, tokenizer, target_tokens, seed=42):
    """
    Randomly sample entries until token count exceeds target.
    
    Args:
        entries: List of all entries
        tokenizer: Tokenizer for counting tokens
        target_tokens: Target token count to reach
        seed: Random seed for reproducibility
    
    Returns:
        sampled_entries: List of sampled entries
        total_tokens: Total tokens in sampled entries
    """
    random.seed(seed)
    
    # Shuffle entries
    shuffled_entries = entries.copy()
    random.shuffle(shuffled_entries)
    
    sampled_entries = []
    total_tokens = 0
    
    print(f"\nSampling entries until reaching {target_tokens:,} tokens...")
    
    for entry in tqdm(shuffled_entries, desc="Sampling"):
        entry_tokens = count_entry_tokens(entry, tokenizer)
        
        sampled_entries.append(entry)
        total_tokens += entry_tokens
        
        if total_tokens >= target_tokens:
            break
    
    return sampled_entries, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Sample ALLaVA instruct entries until target token count")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/media/data2/maschan/internvl/data/allava/allava_instruct_train.jsonl",
        help="Path to ALLaVA instruct training JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/allava/allava_instruct_train_sampled.jsonl",
        help="Output JSONL file path for sampled data"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=14000000,  # 14M tokens
        help="Target token count to reach (default: 14M)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print("Loading InternVL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        trust_remote_code=True,
        use_fast=False
    )
    
    # Load all entries
    entries = load_allava_instruct(args.input_file)
    
    if not entries:
        print("ERROR: No entries loaded!")
        return
    
    # Sample until target tokens
    sampled_entries, total_tokens = sample_until_target_tokens(
        entries,
        tokenizer,
        args.target_tokens,
        seed=args.seed
    )
    
    # Save sampled entries
    print(f"\nSaving {len(sampled_entries):,} sampled entries to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Calculate statistics
    total_entries = len(sampled_entries)
    total_qa_pairs = 0
    for entry in sampled_entries:
        conversations = entry.get('conversations', [])
        total_qa_pairs += len(conversations) // 2
    
    print("\n" + "=" * 80)
    print("SAMPLING RESULTS")
    print("=" * 80)
    print(f"Target tokens: {args.target_tokens:,}")
    print(f"Actual tokens: {total_tokens:,}")
    print(f"Entries sampled: {total_entries:,}")
    print(f"QA pairs: {total_qa_pairs:,}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

