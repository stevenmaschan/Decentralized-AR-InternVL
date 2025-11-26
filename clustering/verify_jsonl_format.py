#!/usr/bin/env python3
"""
Verify that all train.jsonl files are in conversation format.

Expected format:
{
    "id": 0,
    "image": "data/...",
    "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."},
        ...
    ]
}
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def verify_jsonl_format(jsonl_file, max_check=100):
    """
    Verify that a JSONL file is in conversation format.
    
    Returns:
        (is_valid, errors): tuple of (bool, list of error messages)
    """
    errors = []
    checked = 0
    
    try:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                if checked >= max_check:
                    break
                
                try:
                    entry = json.loads(line)
                    
                    # Check required fields
                    if 'id' not in entry:
                        errors.append(f"Line {line_num}: Missing 'id' field")
                    
                    if 'conversations' not in entry:
                        errors.append(f"Line {line_num}: Missing 'conversations' field")
                        continue
                    
                    # Check conversations format
                    conversations = entry['conversations']
                    if not isinstance(conversations, list):
                        errors.append(f"Line {line_num}: 'conversations' must be a list")
                        continue
                    
                    if len(conversations) == 0:
                        errors.append(f"Line {line_num}: 'conversations' list is empty")
                        continue
                    
                    # Check each conversation entry
                    for conv_idx, conv in enumerate(conversations):
                        if not isinstance(conv, dict):
                            errors.append(f"Line {line_num}, conversation {conv_idx}: Not a dict")
                            continue
                        
                        if 'from' not in conv:
                            errors.append(f"Line {line_num}, conversation {conv_idx}: Missing 'from' field")
                        
                        if 'value' not in conv:
                            errors.append(f"Line {line_num}, conversation {conv_idx}: Missing 'value' field")
                        
                        if 'from' in conv and conv['from'] not in ['human', 'gpt', 'system']:
                            errors.append(f"Line {line_num}, conversation {conv_idx}: Invalid 'from' value: {conv.get('from')}")
                    
                    checked += 1
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON: {e}")
                
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Verify JSONL files are in conversation format"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/media/data2/maschan/internvl/data',
        help='Root directory containing dataset JSONL files'
    )
    parser.add_argument(
        '--max-check',
        type=int,
        default=100,
        help='Maximum number of entries to check per file (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VERIFYING JSONL FILES - CONVERSATION FORMAT")
    print("=" * 80)
    
    # Find all train.jsonl files
    data_dir = Path(args.data_dir)
    jsonl_files = {}
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('cluster-'):
            jsonl_file = dataset_dir / 'train.jsonl'
            if jsonl_file.exists():
                dataset_name = dataset_dir.name
                jsonl_files[dataset_name] = jsonl_file
    
    print(f"\nFound {len(jsonl_files)} datasets with train.jsonl files")
    
    # Verify each file
    all_valid = True
    for dataset_name, jsonl_file in sorted(jsonl_files.items()):
        print(f"\nVerifying {dataset_name}...")
        is_valid, errors = verify_jsonl_format(jsonl_file, args.max_check)
        
        if is_valid:
            print(f"  ✓ Valid conversation format (checked {args.max_check} entries)")
        else:
            all_valid = False
            print(f"  ✗ Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"    - {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more errors")
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ All JSONL files are in valid conversation format!")
    else:
        print("✗ Some JSONL files have format errors. Please fix them before partitioning.")
    print("=" * 80)


if __name__ == '__main__':
    main()

