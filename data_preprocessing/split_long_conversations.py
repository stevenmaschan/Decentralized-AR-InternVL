#!/usr/bin/env python3
"""
Split entries with 10 or more QA pairs into multiple entries, each with less than 10 QA pairs.
"""

import json
import argparse
from tqdm import tqdm


def split_conversations(conversations, max_pairs=9):
    """
    Split conversations into chunks, each with at most max_pairs QA pairs.
    Each chunk should start with <image> tag on the first human turn.
    """
    num_pairs = len(conversations) // 2
    if num_pairs <= max_pairs:
        return [conversations]
    
    chunks = []
    current_chunk = []
    pair_count = 0
    
    i = 0
    while i < len(conversations):
        # Check if this is the start of a new pair (human turn)
        if conversations[i].get('from') == 'human':
            # If we've reached max_pairs, start a new chunk
            if pair_count >= max_pairs:
                chunks.append(current_chunk)
                current_chunk = []
                pair_count = 0
            
            # Add human turn
            human_value = conversations[i].get('value', '')
            # Add <image> tag to first human turn of new chunk
            if pair_count == 0 and '<image>' not in human_value:
                human_value = f"<image>\n{human_value}"
            
            current_chunk.append({
                "from": "human",
                "value": human_value
            })
            
            # Add corresponding GPT turn
            if i + 1 < len(conversations):
                current_chunk.append(conversations[i + 1])
                i += 2
            else:
                i += 1
            
            pair_count += 1
        else:
            # Should not happen if format is correct, but handle it
            current_chunk.append(conversations[i])
            i += 1
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def process_dataset(input_file, output_file, max_pairs=9):
    """Process the dataset and split long conversations."""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} entries")
    
    new_data = []
    split_count = 0
    original_count = 0
    
    for entry in tqdm(data, desc="Processing entries"):
        conversations = entry.get('conversations', [])
        num_pairs = len(conversations) // 2
        
        if num_pairs > max_pairs:
            # Split this entry
            original_count += 1
            chunks = split_conversations(conversations, max_pairs)
            
            for chunk in chunks:
                new_entry = {
                    "id": entry.get('id', 0),
                    "image": entry.get('image', ''),
                    "width": entry.get('width', 0),
                    "height": entry.get('height', 0),
                    "conversations": chunk
                }
                new_data.append(new_entry)
                split_count += 1
        else:
            # Keep as is
            new_data.append(entry)
    
    print(f"\nSplit {original_count:,} entries into {split_count:,} entries")
    print(f"Total entries after splitting: {len(new_data):,}")
    
    # Verify all entries have <= max_pairs
    max_pairs_found = 0
    entries_over_limit = 0
    for entry in new_data:
        num_pairs = len(entry.get('conversations', [])) // 2
        if num_pairs > max_pairs_found:
            max_pairs_found = num_pairs
        if num_pairs > max_pairs:
            entries_over_limit += 1
    
    print(f"\nVerification:")
    print(f"  Maximum QA pairs in any entry: {max_pairs_found}")
    print(f"  Entries with > {max_pairs} pairs: {entries_over_limit}")
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Split entries with 10+ QA pairs into multiple entries')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--max_pairs', type=int, default=9,
                        help='Maximum QA pairs per entry (default: 9)')
    
    args = parser.parse_args()
    process_dataset(args.input_file, args.output_file, args.max_pairs)


if __name__ == '__main__':
    main()


