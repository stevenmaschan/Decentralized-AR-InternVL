#!/usr/bin/env python3
"""
Extract CLIP features for all images in DocVQA dataset with multiprocessing support.
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor
from multiprocessing import Process, Queue, Manager
import time
import multiprocessing

# Use spawn method for CUDA multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_clip_model(device='cuda'):
    """Load CLIP ViT-L/14 model with 336x336 input size."""
    print(f"Loading CLIP ViT-L/14 336 model on {device}...", flush=True)
    
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        use_safetensors=True
    ).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    print(f"Model loaded on {device}", flush=True)
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
        print(f"Error processing {image_path}: {e}", flush=True)
        return None


def worker_process(worker_id, device, image_queue, result_queue, images_dir):
    """Worker process that processes images on a specific GPU."""
    # Set CUDA device for this process
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
        torch.cuda.set_device(device_id)
        actual_device = f'cuda:{device_id}'
    else:
        actual_device = device
    
    print(f"Worker {worker_id} starting on {actual_device}...", flush=True)
    
    # Load CLIP model on this GPU
    model, processor = load_clip_model(actual_device)
    
    processed_count = 0
    error_count = 0
    
    while True:
        item = image_queue.get()
        if item is None:  # Poison pill to stop
            break
        
        idx, image_filename = item
        full_image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(full_image_path):
            result_queue.put((idx, None, image_filename, f"File not found"))
            error_count += 1
            continue
        
        features = extract_features_for_image(model, processor, full_image_path, actual_device)
        if features is not None:
            result_queue.put((idx, features[0], image_filename, None))
            processed_count += 1
        else:
            result_queue.put((idx, None, image_filename, "Feature extraction failed"))
            error_count += 1
    
    print(f"Worker {worker_id} finished: {processed_count} processed, {error_count} errors", flush=True)


def extract_features_from_json(
    json_file,
    images_dir,
    output_dir,
    num_gpus=3,
    batch_size=32
):
    """
    Extract CLIP features for all images listed in JSON/JSONL file using multiprocessing.
    
    Args:
        json_file: Path to DocVQA JSON/JSONL file
        images_dir: Directory containing DocVQA images
        output_dir: Directory to save features
        num_gpus: Number of GPUs to use
        batch_size: Batch size (not used in current implementation, kept for compatibility)
    """
    print("Loading JSON/JSONL file...", flush=True)
    
    # Check if it's JSONL or JSON
    is_jsonl = json_file.endswith('.jsonl')
    
    if is_jsonl:
        data = []
        with open(json_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(json_file, 'r') as f:
            data = json.load(f)
    
    # Get unique image filenames with their indices
    image_items = []
    seen = set()
    for idx, entry in enumerate(data):
        image_path = entry.get('image', '')
        if image_path:
            # Extract filename from path (e.g., "data/docvqa/images/ffbf0023_4.png" -> "ffbf0023_4.png")
            image_filename = os.path.basename(image_path)
            if image_filename not in seen:
                seen.add(image_filename)
                image_items.append((len(image_items), image_filename))
    
    print(f"Found {len(image_items):,} unique images in JSON/JSONL", flush=True)
    
    # Create queues
    image_queue = Queue()
    result_queue = Queue()
    
    # Put all images in queue
    for idx, filename in image_items:
        image_queue.put((idx, filename))
    
    # Add poison pills to stop workers
    for _ in range(num_gpus):
        image_queue.put(None)
    
    # Start worker processes
    workers = []
    devices = [f'cuda:{i}' for i in range(num_gpus)]
    
    print(f"Starting {num_gpus} worker processes on {devices}...", flush=True)
    for worker_id, device in enumerate(devices):
        p = Process(target=worker_process, args=(worker_id, device, image_queue, result_queue, images_dir))
        p.start()
        workers.append(p)
    
    # Collect results
    print("Extracting CLIP features...", flush=True)
    results = {}
    total = len(image_items)
    
    with tqdm(total=total, desc="Processing images") as pbar:
        for _ in range(total):
            idx, features, image_filename, error = result_queue.get()
            if features is not None:
                results[idx] = (features, image_filename)
            else:
                if error:
                    print(f"Warning: {image_filename} - {error}", flush=True)
            pbar.update(1)
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    # Sort results by index
    sorted_results = sorted(results.items())
    features_list = [feat for _, (feat, _) in sorted_results]
    valid_paths = [path for _, (_, path) in sorted_results]
    
    if len(features_list) == 0:
        print("No features extracted!", flush=True)
        return
    
    # Stack features
    features_array = np.stack(features_list)
    print(f"Extracted features shape: {features_array.shape}", flush=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset name
    dataset_name = "docvqa"
    
    # Save features as NPZ
    output_file = os.path.join(output_dir, f"{dataset_name}_clip_features.npz")
    print(f"Saving features to {output_file}...", flush=True)
    
    np.savez_compressed(
        output_file,
        features=features_array,
        image_paths=np.array(valid_paths)
    )
    
    # Also save as array for consistency
    array_file = os.path.join(output_dir, f"{dataset_name}_clip_features_array.npy")
    np.save(array_file, features_array)
    print(f"Saved array to {array_file}", flush=True)
    
    # Save metadata
    metadata = {
        "num_samples": len(valid_paths),
        "feature_dim": int(features_array.shape[1]),
        "dataset": dataset_name,
        "model": "openai/clip-vit-large-patch14-336",
        "images_dir": images_dir,
        "json_file": json_file,
        "num_gpus": num_gpus
    }
    
    metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}", flush=True)
    print("Done!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for DocVQA images")
    parser.add_argument(
        "--json_file",
        type=str,
        default="/media/data2/maschan/internvl/data/docvqa/train.jsonl",
        help="Path to DocVQA JSON/JSONL file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/docvqa/images",
        help="Directory containing DocVQA images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/maschan/ddfm/InternVL/clustering/docvqa",
        help="Output directory for features"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=3,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (for compatibility, not currently used)"
    )
    
    args = parser.parse_args()
    
    # Validate GPUs
    if not torch.cuda.is_available():
        print("CUDA not available, cannot use GPUs", flush=True)
        return
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.", flush=True)
        args.num_gpus = available_gpus
    
    extract_features_from_json(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
