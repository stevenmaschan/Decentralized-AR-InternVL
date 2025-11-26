#!/usr/bin/env python3
"""
Rebuild the image path to identity mapping from all unique paths.

This script reads all paths from unique_image_paths.txt, groups them by identity,
and creates a mapping file showing which paths point to the same images.
"""

import os
from collections import defaultdict


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
    import re
    coco_match = re.match(r'COCO_(train|val)(\d{4})_(\d+)$', name_without_ext)
    if coco_match:
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


def main():
    base_dir = "/media/data2/maschan/internvl/data"
    unique_paths_file = os.path.join(base_dir, "unique_image_paths.txt")
    mapping_file = os.path.join(base_dir, "image_path_identity_mapping.txt")
    
    print("Loading all unique image paths...")
    all_paths = []
    with open(unique_paths_file, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if path:
                all_paths.append(path)
    
    print(f"Loaded {len(all_paths):,} paths")
    
    # Group paths by identity
    print("\nGrouping paths by identity...")
    identity_to_paths = defaultdict(set)
    
    for path in all_paths:
        identity = extract_image_identity(path)
        identity_to_paths[identity].add(path)
    
    print(f"Found {len(identity_to_paths):,} unique identities")
    
    # Separate duplicates from singles
    duplicate_groups = {k: v for k, v in identity_to_paths.items() if len(v) > 1}
    single_groups = {k: v for k, v in identity_to_paths.items() if len(v) == 1}
    
    print(f"  - {len(duplicate_groups):,} identities with multiple paths (duplicates)")
    print(f"  - {len(single_groups):,} identities with single path")
    
    # Write mapping file
    print(f"\nWriting mapping file: {mapping_file}")
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("IMAGE PATH TO IDENTITY MAPPING\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file shows which paths point to the same images.\n\n")
        
        if duplicate_groups:
            f.write(f"DUPLICATE IMAGES ({len(duplicate_groups)} groups):\n")
            f.write("-" * 80 + "\n\n")
            
            for identity in sorted(duplicate_groups.keys()):
                paths = sorted(duplicate_groups[identity])
                f.write(f"Identity: {identity}\n")
                for path in paths:
                    f.write(f"  {path}\n")
                f.write("\n")
        else:
            f.write("DUPLICATE IMAGES (0 groups):\n")
            f.write("-" * 80 + "\n\n")
        
        if single_groups:
            f.write(f"\n\nSINGLE IMAGES ({len(single_groups):,} identities):\n")
            f.write("-" * 80 + "\n\n")
            f.write("(These identities have only one path each)\n")
            f.write(f"Total single image identities: {len(single_groups):,}\n")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total paths: {len(all_paths):,}")
    print(f"Total identities: {len(identity_to_paths):,}")
    print(f"Duplicate groups: {len(duplicate_groups):,}")
    print(f"Single identities: {len(single_groups):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()

