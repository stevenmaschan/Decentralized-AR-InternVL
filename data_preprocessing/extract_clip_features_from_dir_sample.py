#!/usr/bin/env python3
"""
Sample random images from a directory and extract CLIP features.

This script:
1. Lists all images in the directory
2. Samples N random images
3. Extracts CLIP features for those images
4. Saves features to disk
"""

import os
import argparse
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPModel, CLIPImageProcessor


def load_clip_model(device='cuda'):
    """Load CLIP ViT-L/14 model with 336x336 input size."""
    print("Loading CLIP ViT-L/14 336 model from HuggingFace...")
    
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        use_safetensors=True
    ).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    print(f"Model loaded on {device}")
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


def get_all_images(images_dir):
    """Get all image files from directory recursively."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_paths = []
    
    print(f"Scanning {images_dir} for images...")
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    return image_paths


def sample_and_extract_features(
    images_dir,
    output_dir,
    num_samples=5000,
    device='cuda',
    seed=42
):
    """
    Sample random images from directory and extract CLIP features.
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save features
        num_samples: Number of random samples
        device: CUDA device
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all images
    all_images = get_all_images(images_dir)
    print(f"Found {len(all_images):,} total images")
    
    if len(all_images) == 0:
        print("No images found!")
        return
    
    # Sample random images
    if len(all_images) < num_samples:
        print(f"Warning: Only {len(all_images):,} images available, using all")
        sampled_images = all_images
    else:
        print(f"Sampling {num_samples:,} random images...")
        sampled_images = random.sample(all_images, num_samples)
    
    print(f"Processing {len(sampled_images):,} images...")
    
    # Load CLIP model
    model, processor = load_clip_model(device)
    
    # Extract features
    print("Extracting CLIP features...")
    features_list = []
    valid_paths = []
    
    for image_path in tqdm(sampled_images):
        features = extract_features_for_image(model, processor, image_path, device)
        if features is not None:
            features_list.append(features[0])  # Remove batch dimension
            valid_paths.append(image_path)
    
    if len(features_list) == 0:
        print("No features extracted!")
        return
    
    # Stack features
    features_array = np.stack(features_list)
    print(f"Extracted features shape: {features_array.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset name from output_dir
    dataset_name = os.path.basename(output_dir.rstrip('/'))
    if dataset_name == '':
        dataset_name = os.path.basename(os.path.dirname(output_dir.rstrip('/')))
    
    # Save features
    output_file = os.path.join(output_dir, f"{dataset_name}_clip_features.npz")
    print(f"Saving features to {output_file}...")
    
    # Store relative paths
    rel_paths = [os.path.relpath(p, images_dir) for p in valid_paths]
    
    np.savez_compressed(
        output_file,
        features=features_array,
        image_paths=rel_paths
    )
    
    # Save metadata
    import json
    metadata = {
        "num_samples": len(valid_paths),
        "feature_dim": features_array.shape[1],
        "dataset": dataset_name,
        "model": "openai/clip-vit-large-patch14-336",
        "seed": seed,
        "images_dir": images_dir
    }
    
    metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Sample and extract CLIP features from images directory")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for features (default: clustering/{dataset_name})"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of random samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device (e.g., cuda, cuda:0, cuda:1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set default output_dir based on images_dir if not provided
    if args.output_dir is None:
        dataset_name = os.path.basename(args.images_dir.rstrip('/'))
        if dataset_name == '':
            dataset_name = os.path.basename(os.path.dirname(args.images_dir.rstrip('/')))
        args.output_dir = os.path.join("/home/maschan/ddfm/InternVL/clustering", dataset_name)
    
    # Validate CUDA device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            args.device = 'cpu'
        elif ':' in args.device:
            device_id = int(args.device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"Invalid device {args.device}, using cuda:0")
                args.device = 'cuda:0'
    
    sample_and_extract_features(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

