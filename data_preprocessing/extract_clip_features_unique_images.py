#!/usr/bin/env python3
"""
Extract CLIP features for all unique images.

This script:
1. Loads unique image paths from unique_image_paths.txt
2. Groups paths by identity (same underlying image)
3. Extracts CLIP features for each unique identity
4. Creates a consolidated array where each row corresponds to a unique image identity
"""

import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
import argparse


def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load CLIP ViT-L/14 336 model."""
    print(f"Loading CLIP model on {device}...")
    model_name = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True
    ).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(model_name)
    print("CLIP model loaded successfully")
    return model, processor


def extract_features_for_image(model, processor, image_path, device):
    """Extract CLIP features for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def extract_image_identity(image_path):
    """
    Extract a normalized identity from an image path.
    Must match the logic used in add_allava_to_unique_images.py
    """
    import re
    
    # Remove data/ prefix if present
    path = image_path.replace('data/', '')
    
    # Extract filename without extension
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Handle COCO images: COCO_train2014_000000000009.jpg -> coco_000000000009
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


def load_unique_image_identities(identities_file):
    """Load unique image identities from file."""
    identities = []
    with open(identities_file, 'r') as f:
        for line in f:
            identity = line.strip()
            if identity:
                identities.append(identity)
    return sorted(identities)


def load_unique_paths_grouped_by_identity(paths_file):
    """
    Load unique image paths and group them by identity.
    Returns:
        identity_to_paths: dict mapping identity to list of paths
        identity_to_primary_path: dict mapping identity to a primary path (first one)
    """
    identity_to_paths = defaultdict(list)
    
    with open(paths_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                identity = extract_image_identity(path)
                identity_to_paths[identity].append(path)
    
    # Create primary path mapping (use first path for each identity)
    identity_to_primary_path = {}
    for identity, paths in identity_to_paths.items():
        identity_to_primary_path[identity] = paths[0]
    
    return dict(identity_to_paths), identity_to_primary_path


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for all unique images")
    parser.add_argument(
        "--unique_paths_file",
        type=str,
        default="/media/data2/maschan/internvl/data/unique_image_paths.txt",
        help="Path to unique_image_paths.txt"
    )
    parser.add_argument(
        "--unique_identities_file",
        type=str,
        default="/media/data2/maschan/internvl/data/unique_image_paths_normalized.txt",
        help="Path to unique_image_paths_normalized.txt"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/data2/maschan/internvl/data",
        help="Root directory for image data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clustering/consolidated_clip_features.npy",
        help="Output npy file for consolidated features"
    )
    parser.add_argument(
        "--output_paths_file",
        type=str,
        default="clustering/consolidated_clip_features_paths.txt",
        help="Output file for paths (one per line, matching array rows)"
    )
    parser.add_argument(
        "--output_metadata_file",
        type=str,
        default="clustering/consolidated_clip_features_metadata.json",
        help="Output metadata JSON file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (for resuming)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 80)
    print("EXTRACTING CLIP FEATURES FOR UNIQUE IMAGES")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Load unique identities
    print(f"\nLoading unique image identities from {args.unique_identities_file}...")
    unique_identities = load_unique_image_identities(args.unique_identities_file)
    print(f"Found {len(unique_identities):,} unique image identities")
    
    # Load paths grouped by identity
    print(f"\nLoading unique image paths from {args.unique_paths_file}...")
    identity_to_paths, identity_to_primary_path = load_unique_paths_grouped_by_identity(args.unique_paths_file)
    print(f"Found {len(identity_to_paths):,} unique identities with paths")
    
    # Create identity to index mapping
    identity_to_index = {identity: idx for idx, identity in enumerate(unique_identities)}
    
    # Limit number of images if specified
    if args.max_images:
        unique_identities = unique_identities[:args.max_images]
        print(f"Limited to first {len(unique_identities):,} identities")
    
    # Initialize output array
    num_identities = len(unique_identities)
    feature_dim = 768  # CLIP ViT-L/14 feature dimension
    
    # Load existing features if resuming
    if args.start_idx > 0 and os.path.exists(args.output_file):
        print(f"Loading existing features from {args.output_file} (resuming from index {args.start_idx})...")
        existing_features = np.load(args.output_file)
        consolidated_features = existing_features.copy()
        identity_has_features = np.any(consolidated_features != 0, axis=1)  # Check which have non-zero features
        print(f"Loaded existing features, shape: {consolidated_features.shape}")
    else:
        consolidated_features = np.zeros((num_identities, feature_dim), dtype=np.float32)
        identity_has_features = np.zeros(num_identities, dtype=bool)
    
    # Load CLIP model
    model, processor = load_clip_model(device)
    
    # Process images
    print(f"\nExtracting features for {num_identities:,} unique images...")
    print(f"Starting from index {args.start_idx}")
    
    processed_count = 0
    failed_count = 0
    
    for idx in tqdm(range(args.start_idx, num_identities), desc="Extracting features"):
        identity = unique_identities[idx]
        
        # Get primary path for this identity
        primary_path = identity_to_primary_path.get(identity)
        
        if not primary_path:
            print(f"Warning: No path found for identity {identity}")
            failed_count += 1
            continue
        
        # Construct full image path
        if os.path.isabs(primary_path):
            full_image_path = primary_path
        elif primary_path.startswith('data/'):
            # Path already includes data/ prefix, need to prepend root
            # data_root should be /media/data2/maschan/internvl/data
            # and path is data/ai2diagram/images/10.png
            # So we need /media/data2/maschan/internvl/data/ai2diagram/images/10.png
            # Or simpler: if data_root ends with /data, just join
            if args.data_root.endswith('/data'):
                full_image_path = os.path.join(args.data_root, primary_path[5:])  # Remove 'data/'
            else:
                full_image_path = os.path.join(args.data_root, primary_path)
        else:
            full_image_path = os.path.join(args.data_root, primary_path)
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            failed_count += 1
            continue
        
        # Extract features
        features = extract_features_for_image(model, processor, full_image_path, device)
        
        if features is not None:
            consolidated_features[idx] = features[0]  # Remove batch dimension
            identity_has_features[idx] = True
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\nProcessed: {processed_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Total: {processed_count + failed_count:,}")
    
    # Save consolidated features
    print(f"\nSaving consolidated features to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, consolidated_features)
    
    # Save paths (order matching the features array)
    print(f"Saving paths to {args.output_paths_file}...")
    with open(args.output_paths_file, 'w') as f:
        for identity in unique_identities:
            path = identity_to_primary_path.get(identity, '')
            f.write(f"{path}\n")
    
    # Save metadata
    print(f"Saving metadata to {args.output_metadata_file}...")
    metadata = {
        'num_identities': num_identities,
        'feature_dim': feature_dim,
        'identities_with_features': int(np.sum(identity_has_features)),
        'identities_without_features': int(np.sum(~identity_has_features)),
        'model': 'openai/clip-vit-large-patch14-336',
        'input_size': 336,
    }
    with open(args.output_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique identities: {num_identities:,}")
    print(f"Identities with features: {np.sum(identity_has_features):,}")
    print(f"Identities without features: {np.sum(~identity_has_features):,}")
    if num_identities > 0:
        print(f"Coverage: {np.sum(identity_has_features) / num_identities * 100:.2f}%")
    print(f"Output file: {args.output_file}")
    print(f"Output shape: {consolidated_features.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()

