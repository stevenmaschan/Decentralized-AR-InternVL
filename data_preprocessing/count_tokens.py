#!/usr/bin/env python3
"""
Count tokens in RefCOCO datasets.
"""

import json
import argparse
from tqdm import tqdm

try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    try:
        from transformers import AutoTokenizer
        USE_TIKTOKEN = False
    except ImportError:
        print("Error: Need either tiktoken or transformers library")
        exit(1)


def get_tokenizer():
    """Get tokenizer for counting tokens."""
    if USE_TIKTOKEN:
        # Use GPT-4 tokenizer (cl100k_base)
        return tiktoken.get_encoding("cl100k_base")
    else:
        # Use LLaMA tokenizer as fallback
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return tokenizer


def count_tokens_in_text(text, tokenizer):
    """Count tokens in text."""
    if USE_TIKTOKEN:
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.encode(text, add_special_tokens=False))


def count_tokens_in_entry(entry, tokenizer):
    """Count tokens in a single entry."""
    total_tokens = 0
    for conv in entry.get('conversations', []):
        value = conv.get('value', '')
        tokens = count_tokens_in_text(value, tokenizer)
        total_tokens += tokens
    return total_tokens


def process_dataset(input_file, tokenizer_name=None):
    """Count tokens in dataset."""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} entries")
    
    # Initialize tokenizer
    if tokenizer_name:
        if USE_TIKTOKEN:
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = get_tokenizer()
    
    print(f"Using tokenizer: {tokenizer_name if tokenizer_name else ('tiktoken cl100k_base' if USE_TIKTOKEN else 'transformers')}")
    print()
    
    # Count tokens
    total_tokens = 0
    tokens_per_entry = []
    
    for entry in tqdm(data, desc="Counting tokens"):
        tokens = count_tokens_in_entry(entry, tokenizer)
        tokens_per_entry.append(tokens)
        total_tokens += tokens
    
    # Statistics
    import statistics
    avg_tokens = statistics.mean(tokens_per_entry)
    median_tokens = statistics.median(tokens_per_entry)
    min_tokens = min(tokens_per_entry)
    max_tokens = max(tokens_per_entry)
    
    print()
    print("="*80)
    print("TOKEN COUNT STATISTICS")
    print("="*80)
    print(f"Total entries: {len(data):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per entry: {avg_tokens:.1f}")
    print(f"Median tokens per entry: {median_tokens:.1f}")
    print(f"Min tokens per entry: {min_tokens:,}")
    print(f"Max tokens per entry: {max_tokens:,}")
    print("="*80)
    
    return {
        'total_entries': len(data),
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'median_tokens': median_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens
    }


def main():
    parser = argparse.ArgumentParser(description='Count tokens in RefCOCO datasets')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Tokenizer name (for tiktoken: cl100k_base, p50k_base, etc. For transformers: model name)')
    
    args = parser.parse_args()
    process_dataset(args.input_file, args.tokenizer)


if __name__ == '__main__':
    main()


