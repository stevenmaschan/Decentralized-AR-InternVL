#!/usr/bin/env python3
"""
Add ALLaVA sampled images to unique images analysis.

This script:
1. Extracts image paths from ALLaVA sampled file
2. Adds them to unique_image_paths.txt
3. Updates unique_image_paths_normalized.txt
4. Updates image_path_identity_mapping.txt
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path


def extract_image_identity(image_path):
    """
    Extract a normalized identity from an image path.
    Handles different path formats to identify same underlying images.
    """
    # Remove data/ prefix if present
    path = image_path.replace('data/', '')
    
    # Extract filename without extension
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Handle COCO images: COCO_train2014_000000000009.jpg -> coco_000000000009
    coco_match = re.match(r'COCO_(train|val)(\d{4})_(\d+)$', name_without_ext)
    if coco_match:
        year = coco_match.group(2)
        img_id = coco_match.group(3)
        return f"coco_{img_id}"
    
    # Handle COCO without prefix: 000000000009.jpg -> coco_000000000009
    if re.match(r'^\d{12}$', name_without_ext):
        return f"coco_{name_without_ext}"
    
    # Handle TextCaps/TextVQA: filename is the identity
    if 'textcaps' in path.lower() or 'textvqa' in path.lower():
        return f"textcaps_textvqa_{filename}"
    
    # For other images, use dataset name + filename as identity
    # Extract dataset name from path
    parts = path.split('/')
    if len(parts) >= 2:
        dataset = parts[0]
        return f"{dataset}_{filename}"
    
    return f"unknown_{filename}"


def load_existing_paths(unique_paths_file):
    """Load existing unique image paths."""
    if not os.path.exists(unique_paths_file):
        return set()
    
    paths = set()
    with open(unique_paths_file, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if path:
                paths.add(path)
    return paths


def load_existing_identities(normalized_file):
    """Load existing normalized identities."""
    if not os.path.exists(normalized_file):
        return set()
    
    identities = set()
    with open(normalized_file, 'r', encoding='utf-8') as f:
        for line in f:
            identity = line.strip()
            if identity:
                identities.add(identity)
    return identities


def load_existing_mapping(mapping_file):
    """Load existing path to identity mapping."""
    if not os.path.exists(mapping_file):
        return defaultdict(set)
    
    identity_to_paths = defaultdict(set)
    current_identity = None
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Identity: '):
                current_identity = line.replace('Identity: ', '').strip()
            elif line.startswith('  data/'):
                path = line.strip()
                if current_identity:
                    identity_to_paths[current_identity].add(path)
    
    return identity_to_paths


def extract_allava_paths(allava_file):
    """Extract image paths from ALLaVA sampled file."""
    paths = []
    print(f"Loading ALLaVA sampled file: {allava_file}")
    
    with open(allava_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                image_path = entry.get('image', '')
                if image_path:
                    paths.append(image_path)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"Extracted {len(paths):,} image paths from ALLaVA")
    return paths


def main():
    base_dir = "/media/data2/maschan/internvl/data"
    allava_file = os.path.join(base_dir, "allava/allava_instruct_train_sampled.jsonl")
    
    unique_paths_file = os.path.join(base_dir, "unique_image_paths.txt")
    normalized_file = os.path.join(base_dir, "unique_image_paths_normalized.txt")
    mapping_file = os.path.join(base_dir, "image_path_identity_mapping.txt")
    
    # Extract ALLaVA paths
    allava_paths = extract_allava_paths(allava_file)
    
    if not allava_paths:
        print("ERROR: No paths extracted from ALLaVA file!")
        return
    
    # Load existing data
    print("\nLoading existing unique paths...")
    existing_paths = load_existing_paths(unique_paths_file)
    print(f"Found {len(existing_paths):,} existing unique paths")
    
    print("\nLoading existing normalized identities...")
    existing_identities = load_existing_identities(normalized_file)
    print(f"Found {len(existing_identities):,} existing identities")
    
    print("\nLoading existing path-to-identity mapping...")
    identity_to_paths = load_existing_mapping(mapping_file)
    print(f"Found {len(identity_to_paths):,} existing identity groups")
    
    # Process ALLaVA paths
    print("\nProcessing ALLaVA paths...")
    new_paths = set()
    allava_identity_to_paths = defaultdict(set)
    
    for path in allava_paths:
        if path not in existing_paths:
            new_paths.add(path)
            identity = extract_image_identity(path)
            allava_identity_to_paths[identity].add(path)
    
    print(f"Found {len(new_paths):,} new paths (not in existing)")
    print(f"Found {len(allava_identity_to_paths):,} unique identities in ALLaVA")
    
    # Merge with existing identities
    new_identities = set()
    for identity in allava_identity_to_paths.keys():
        if identity not in existing_identities:
            new_identities.add(identity)
        # Add paths to existing identity groups if identity already exists
        if identity in identity_to_paths:
            identity_to_paths[identity].update(allava_identity_to_paths[identity])
        else:
            identity_to_paths[identity] = allava_identity_to_paths[identity]
    
    print(f"Found {len(new_identities):,} new identities")
    
    # Update unique_image_paths.txt
    print(f"\nUpdating {unique_paths_file}...")
    all_paths = existing_paths | new_paths
    
    with open(unique_paths_file, 'w', encoding='utf-8') as f:
        for path in sorted(all_paths):
            f.write(path + '\n')
    
    print(f"Updated: {len(all_paths):,} total paths ({len(new_paths):,} added)")
    
    # Update unique_image_paths_normalized.txt
    print(f"\nUpdating {normalized_file}...")
    all_identities = existing_identities | new_identities
    
    with open(normalized_file, 'w', encoding='utf-8') as f:
        for identity in sorted(all_identities):
            f.write(identity + '\n')
    
    print(f"Updated: {len(all_identities):,} total identities ({len(new_identities):,} added)")
    
    # Update image_path_identity_mapping.txt
    print(f"\nUpdating {mapping_file}...")
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("IMAGE PATH TO IDENTITY MAPPING\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file shows which paths point to the same images.\n\n")
        
        # Group by whether identity has multiple paths
        duplicate_groups = {k: v for k, v in identity_to_paths.items() if len(v) > 1}
        single_groups = {k: v for k, v in identity_to_paths.items() if len(v) == 1}
        
        f.write(f"DUPLICATE IMAGES ({len(duplicate_groups)} groups):\n")
        f.write("-" * 80 + "\n\n")
        
        for identity in sorted(duplicate_groups.keys()):
            paths = sorted(duplicate_groups[identity])
            f.write(f"Identity: {identity}\n")
            for path in paths:
                f.write(f"  {path}\n")
            f.write("\n")
        
        if single_groups:
            f.write(f"\n\nSINGLE IMAGES ({len(single_groups)} identities):\n")
            f.write("-" * 80 + "\n\n")
            f.write("(These identities have only one path each)\n\n")
            # Optionally list them, but might be too many
            # For now, just mention the count
    
    print(f"Updated mapping: {len(identity_to_paths):,} identity groups")
    print(f"  - {len(duplicate_groups):,} groups with duplicates")
    print(f"  - {len(single_groups):,} groups with single images")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique paths: {len(all_paths):,}")
    print(f"Total unique identities: {len(all_identities):,}")
    print(f"ALLaVA paths added: {len(new_paths):,}")
    print(f"ALLaVA new identities: {len(new_identities):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()

