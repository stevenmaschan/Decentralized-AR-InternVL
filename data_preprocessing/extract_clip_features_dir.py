#!/usr/bin/env python3
"""
Extract CLIP ViT-L/14 336 features from all images in a directory.

This script:
1. Loads CLIP ViT-L/14 model with 336x336 input size
2. Scans a directory for all images
3. Extracts visual features from all images found
4. Saves features to disk (numpy format)
"""

import os
import json
import argparse
import numpy as np
import torch
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


def get_image_paths_from_directory(images_dir):
    """Get all image paths from directory recursively."""
    print(f"Scanning images directory: {images_dir}...")
    image_paths = []
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    extensions += [ext.upper() for ext in extensions]
    
    # Scan directory recursively
    for ext in extensions:
        image_paths.extend(Path(images_dir).rglob(f'*{ext}'))
    
    image_paths = sorted([str(p) for p in image_paths])
    print(f"Found {len(image_paths)} images")
    
    return image_paths


def extract_features_for_directory(
    dataset_name,
    images_dir,
    output_dir=None,
    device='cuda',
    max_images=None
):
    """
    Extract CLIP features for all images in a directory.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'allava')
        images_dir: Directory containing images
        output_dir: Directory to save features (default: clustering/{dataset_name})
        device: Device to use ('cuda' or 'cpu')
        max_images: Maximum number of images to process (None for all)
    """
    
    # Setup paths
    if output_dir is None:
        # Default: save in clustering directory in InternVL workspace
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(script_dir)
        output_dir = os.path.join(workspace_root, 'clustering', dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model, processor = load_clip_model(device=device)
    
    # Get feature dimension from model
    dummy_image = Image.new('RGB', (336, 336))
    dummy_inputs = processor(images=dummy_image, return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_features = model.get_image_features(**dummy_inputs)
        feature_dim = dummy_features.shape[-1]
    print(f"Feature dimension: {feature_dim}")
    
    # Get image paths
    image_paths = get_image_paths_from_directory(images_dir)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Processing {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Exiting.")
        return
    
    # Process images
    features_dict = {}
    failed_images = []
    
    print("\nExtracting features...")
    for image_path in tqdm(image_paths, desc="Processing images"):
        features = extract_features_for_image(model, processor, image_path, device)
        
        if features is not None:
            # Use relative path as key
            rel_path = os.path.relpath(image_path, images_dir)
            features_dict[rel_path] = features
        else:
            failed_images.append(image_path)
    
    # Save features
    print(f"\nSaving features...")
    
    # Save as numpy archive
    features_file = os.path.join(output_dir, f'{dataset_name}_clip_features.npz')
    np.savez_compressed(features_file, **features_dict)
    print(f"Saved features to {features_file}")
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'model': 'CLIP ViT-L/14',
        'input_size': 336,
        'feature_dim': feature_dim,
        'num_images': len(features_dict),
        'failed_images': failed_images,
        'image_paths': list(features_dict.keys())
    }
    
    metadata_file = os.path.join(output_dir, f'{dataset_name}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    print(f"\nDone!")
    print(f"Successfully processed: {len(features_dict)} images")
    if failed_images:
        print(f"Failed images: {len(failed_images)}")
        print(f"Failed image list saved in metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-L/14 336 features from all images in a directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'allava')"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images (will scan recursively)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save features (default: clustering/{dataset_name})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = 'cpu'
        else:
            # Validate device number if specified
            if ':' in args.device:
                device_num = int(args.device.split(':')[1])
                if device_num >= torch.cuda.device_count():
                    print(f"CUDA device {device_num} not available, using cuda:0")
                    args.device = 'cuda:0'
    
    extract_features_for_directory(
        dataset_name=args.dataset_name,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()



