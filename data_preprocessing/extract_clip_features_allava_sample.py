#!/usr/bin/env python3
"""
Extract CLIP features for ALLaVA sampled dataset.

This script:
1. Loads ALLaVA sampled JSONL file
2. Extracts CLIP features for each image in order
3. Saves as NPZ file with features array and paths/metadata
"""

import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor
import argparse


def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load CLIP ViT-L/14 model with 336x336 input size."""
    print(f"Loading CLIP ViT-L/14 336 model on {device}...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        use_safetensors=True
    ).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    print("CLIP model loaded successfully")
    return model, processor


def extract_features_for_image(model, processor, image_path, device='cuda'):
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


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for ALLaVA sample")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/media/data2/maschan/internvl/data/allava/allava_instruct_train_sampled.jsonl",
        help="Path to ALLaVA sampled JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clustering/allava/allava_sampled_clip_features.npz",
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/data2/maschan/internvl/data",
        help="Root directory for image data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified."
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 80)
    print("EXTRACTING CLIP FEATURES FOR ALLaVA SAMPLE")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {device}")
    
    # Load ALLaVA sampled file
    print(f"\nLoading ALLaVA sampled file...")
    entries = []
    image_paths = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
                
                # Extract image path
        image_path = entry.get('image', '')
        if image_path:
                    image_paths.append(image_path)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(entries):,} entries")
    print(f"Found {len(image_paths):,} image paths")
    
    # Load CLIP model
    model, processor = load_clip_model(device)
    
    # Extract features (preserve order)
    print(f"\nExtracting CLIP features for {len(image_paths):,} images...")
    features_list = []
    valid_paths = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting features")):
        # Construct full image path
        if os.path.isabs(image_path):
            full_image_path = image_path
        elif image_path.startswith('data/'):
            # Remove 'data/' prefix and join with data_root
            if args.data_root.endswith('/data'):
                full_image_path = os.path.join(args.data_root, image_path[5:])
            else:
                full_image_path = os.path.join(args.data_root, image_path)
        else:
            full_image_path = os.path.join(args.data_root, image_path)
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            # Still append zero features to maintain order
            features_list.append(np.zeros((1, 768), dtype=np.float32))
            valid_paths.append(image_path)
            continue
        
        # Extract features
        features = extract_features_for_image(model, processor, full_image_path, device)
        
        if features is not None:
            features_list.append(features)
            valid_paths.append(image_path)
        else:
            # Append zero features to maintain order
            features_list.append(np.zeros((1, 768), dtype=np.float32))
            valid_paths.append(image_path)
    
    # Stack all features into a single array
    print(f"\nStacking features...")
    features_array = np.vstack(features_list)
    print(f"Features array shape: {features_array.shape}")
    
    # Convert paths to numpy array (for NPZ compatibility)
    image_paths_array = np.array(valid_paths, dtype=object)
    
    # Save as NPZ
    print(f"\nSaving to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    np.savez(
        args.output_file,
        features=features_array,
        image_paths=image_paths_array,
    )
    
    # Save metadata
    metadata = {
        'num_samples': len(features_array),
        'feature_dim': features_array.shape[1],
        'model': 'openai/clip-vit-large-patch14-336',
        'input_size': 336,
        'input_file': args.input_file,
    }
    
    metadata_file = args.output_file.replace('.npz', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries: {len(entries):,}")
    print(f"Features extracted: {len(features_array):,}")
    print(f"Features shape: {features_array.shape}")
    print(f"Output file: {args.output_file}")
    print(f"Metadata file: {metadata_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
