#!/usr/bin/env python3
"""
Extract CLIP ViT-L/14 336 features from images in a dataset.

This script:
1. Loads CLIP ViT-L/14 model with 336x336 input size
2. Processes all images in the specified dataset
3. Extracts visual features
4. Saves features to disk (numpy format)

Usage example for ai2d dataset:
    python extract_clip_features.py \
        --dataset_name ai2d \
        --images_dir /media/data2/maschan/internvl/datasets/ai2d/images \
        --json_file /media/data2/maschan/internvl/datasets/ai2d/ai2d_train.json \
        --device cuda

Or without JSON file (scans images directory):
    python extract_clip_features.py \
        --dataset_name ai2d \
        --images_dir /media/data2/maschan/internvl/datasets/ai2d/images \
        --device cuda
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


def get_image_paths_from_json(json_file, images_dir):
    """Extract image paths from dataset JSON file. Only processes images in JSON entries."""
    image_paths = []
    
    if json_file and os.path.exists(json_file):
        print(f"Reading image paths from {json_file}...")
        # Handle both JSON and JSONL formats
        if json_file.endswith('.jsonl'):
            data = []
            with open(json_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            with open(json_file, 'r') as f:
                data = json.load(f)
        
        found_count = 0
        missing_count = 0
        
        for entry in data:
            image_path = entry.get('image', '')
            if image_path:
                # Handle both absolute and relative paths
                if os.path.isabs(image_path):
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        found_count += 1
                    else:
                        missing_count += 1
                else:
                    full_path = os.path.join(images_dir, image_path)
                    if os.path.exists(full_path):
                        image_paths.append(full_path)
                        found_count += 1
                    else:
                        missing_count += 1
        
        print(f"Found {found_count} images from JSON entries")
        if missing_count > 0:
            print(f"Warning: {missing_count} images from JSON entries not found on disk")
    else:
        # If no JSON file, scan images directory
        print(f"Warning: No JSON file provided. Scanning images directory: {images_dir}...")
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_paths.extend(Path(images_dir).glob(f'*{ext}'))
            image_paths.extend(Path(images_dir).glob(f'*{ext.upper()}'))
        image_paths = [str(p) for p in image_paths]
        print(f"Found {len(image_paths)} images in directory")
    
    return sorted(list(set(image_paths)))  # Remove duplicates and sort


def extract_features_for_dataset(
    dataset_name,
    images_dir,
    json_file=None,
    output_dir=None,
    batch_size=32,
    device='cuda',
    max_images=None
):
    """
    Extract CLIP features for all images in a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ai2d')
        images_dir: Directory containing images
        json_file: Optional JSON file with image paths
        output_dir: Directory to save features (default: images_dir/../features)
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        max_images: Maximum number of images to process (None for all)
    """
    
    # Setup paths
    if output_dir is None:
        # Default: save in clustering directory in InternVL workspace
        # Get the InternVL workspace root (parent of data_preprocessing)
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
    image_paths = get_image_paths_from_json(json_file, images_dir)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
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
            # Use relative path or filename as key
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
        description="Extract CLIP ViT-L/14 336 features from dataset images"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'ai2d')"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Optional JSON file with image paths (e.g., ai2d_train.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save features (default: ../clustering/{dataset_name})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (currently not used, processes one by one)"
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
    
    extract_features_for_dataset(
        dataset_name=args.dataset_name,
        images_dir=args.images_dir,
        json_file=args.json_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

