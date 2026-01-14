#!/usr/bin/env python3
"""
Generic script to split any dataset into clusters for multi-expert evaluation.
Supports both two-stage clustering (fine → coarse) and single-stage clustering.
Uses parallel CLIP feature extraction across multiple GPUs, then sequential cluster assignment.

Usage:
    # Single-stage balanced k-means:
    python clustering/split_dataset.py \
        --input-jsonl data/ai2diagram/test_vlmevalkit.jsonl \
        --output-dir data/ai2diagram/test_2experts \
        --clustering-results-dir clustering/balanced-kmeans_vit-base-patch-16_2-coarse \
        --images-dir data/ai2diagram/AI2D_TEST \
        --device cuda:0 \
        --num-gpus 4 \
        --output-prefix ai2d_test \
        --clip-model openai/clip-vit-base-patch16

    # Two-stage clustering:
    python clustering/split_dataset.py \
        --input-jsonl data/docvqa/val.jsonl \
        --output-dir data/docvqa/val_2experts \
        --clustering-results-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \
        --images-dir data/docvqa/val \
        --device cuda:0 \
        --num-gpus 4 \
        --output-prefix docvqa_val \
        --clip-model openai/clip-vit-base-patch16
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
from multiprocessing import Process, Manager
import os

# Allow loading truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_clustering_results(clustering_results_dir, prefix="clustering"):
    """
    Load clustering results, automatically detecting two-stage or single-stage.
    
    Returns:
        clustering_type: 'two_stage' or 'single_stage'
        n_clusters: number of final clusters
        assign_func: function to assign features to clusters
    """
    clustering_dir = Path(clustering_results_dir)
    
    # Check for two-stage clustering files
    fine_centroids_file = clustering_dir / f"{prefix}_fine_centroids.npy"
    coarse_centroids_file = clustering_dir / f"{prefix}_coarse_centroids.npy"
    fine_to_coarse_file = clustering_dir / f"{prefix}_fine_to_coarse_mapping.npy"
    
    # Check for single-stage clustering files
    centroids_file = clustering_dir / f"{prefix}_centroids.npy"
    
    if fine_centroids_file.exists() and coarse_centroids_file.exists() and fine_to_coarse_file.exists():
        # Two-stage clustering
        print("Detected two-stage clustering results")
        fine_centroids = np.load(fine_centroids_file)
        coarse_centroids = np.load(coarse_centroids_file)
        fine_to_coarse = np.load(fine_to_coarse_file)
        
        print(f"Loaded fine centroids: {fine_centroids.shape}")
        print(f"Loaded coarse centroids: {coarse_centroids.shape}")
        print(f"Loaded fine-to-coarse mapping: {fine_to_coarse.shape}")
        
        n_clusters = len(coarse_centroids)
        
        def assign_func(features):
            """Assign features using two-stage clustering."""
            # Normalize features
            features_norm = features / np.linalg.norm(features)
            
            # Assign to fine cluster
            fine_centroids_norm = fine_centroids / np.linalg.norm(fine_centroids, axis=1, keepdims=True)
            similarities = np.dot(fine_centroids_norm, features_norm)
            fine_cluster_id = np.argmax(similarities)
            
            # Map to coarse cluster
            coarse_cluster_id = int(fine_to_coarse[fine_cluster_id])
            return coarse_cluster_id
        
        return 'two_stage', n_clusters, assign_func
        
    elif centroids_file.exists():
        # Single-stage clustering
        print("Detected single-stage clustering results")
        centroids = np.load(centroids_file)
        print(f"Loaded centroids: {centroids.shape}")
        
        n_clusters = len(centroids)
        
        def assign_func(features):
            """Assign features using single-stage clustering."""
            # Normalize features
            features_norm = features / np.linalg.norm(features)
            
            # Normalize centroids
            centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(centroids_norm, features_norm)
            
            # Assign to cluster with highest similarity
            cluster_id = np.argmax(similarities)
            return int(cluster_id)
        
        return 'single_stage', n_clusters, assign_func
        
    else:
        raise FileNotFoundError(
            f"Could not find clustering results in {clustering_dir}.\n"
            f"Expected either:\n"
            f"  - Two-stage: {prefix}_fine_centroids.npy, {prefix}_coarse_centroids.npy, {prefix}_fine_to_coarse_mapping.npy\n"
            f"  - Single-stage: {prefix}_centroids.npy"
        )


def load_clip_model(device, clip_model_name="openai/clip-vit-base-patch16"):
    """Load CLIP model and processor."""
    print(f"Loading CLIP model: {clip_model_name} on {device}...", flush=True)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    model.eval()
    print(f"Model loaded on {device}", flush=True)
    return model, processor


def extract_features_for_image(clip_model, clip_processor, image_path, device):
    """Extract CLIP features for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into clusters for multi-expert evaluation. '
                    'Supports both two-stage and single-stage clustering results.'
    )
    parser.add_argument('--input-jsonl', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for cluster files')
    parser.add_argument('--clustering-results-dir', type=str, required=True, 
                       help='Directory with clustering results (auto-detects two-stage or single-stage)')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Base device (e.g., cuda:0)')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use for parallel feature extraction')
    parser.add_argument('--output-prefix', type=str, default=None, 
                       help='Prefix for output files (default: input filename stem)')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch16', 
                       help='CLIP model to use (e.g., openai/clip-vit-base-patch16, openai/clip-vit-large-patch14)')
    parser.add_argument('--prefix', type=str, default='clustering',
                       help='Prefix for clustering result files (default: clustering)')
    
    args = parser.parse_args()
    
    # Step 1: Load clustering results
    print("\n" + "=" * 80)
    print("Step 1: Loading clustering results")
    print("=" * 80)
    clustering_type, n_clusters, assign_func = load_clustering_results(
        args.clustering_results_dir, prefix=args.prefix
    )
    print(f"Clustering type: {clustering_type}")
    print(f"Number of clusters: {n_clusters}")
    
    # Step 2: Determine GPU devices
    print("\n" + "=" * 80)
    print("Step 2: Setting up GPU devices")
    print("=" * 80)
    if args.num_gpus > 1:
        base_device_num = int(args.device.split(':')[1]) if ':' in args.device else 0
        devices = [f'cuda:{base_device_num + i}' for i in range(args.num_gpus)]
        print(f"Using {args.num_gpus} GPUs: {devices}")
    else:
        devices = [args.device]
        print(f"Using single device: {devices[0]}")
    
    # Step 3: Load JSONL file
    print("\n" + "=" * 80)
    print("Step 3: Loading JSONL file")
    print("=" * 80)
    input_file = Path(args.input_jsonl)
    if not input_file.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_file}")
    
    entries = []
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc="Loading JSONL", total=total_lines, unit="line"):
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries from {input_file}")
    
    # Step 4: Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = input_file.stem
    
    # Step 5: Extract CLIP features in parallel
    print("\n" + "=" * 80)
    print("Step 5a: Extracting CLIP features for all entries (parallelized, preserving order)")
    print("=" * 80)
    
    images_dir = Path(args.images_dir)
    
    if args.num_gpus > 1:
        # Split entries across GPUs for feature extraction
        chunk_size = len(entries) // args.num_gpus
        remainder = len(entries) % args.num_gpus
        entry_chunks = []
        start_idx = 0
        for i in range(args.num_gpus):
            # Distribute remainder across first few chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            entry_chunks.append(entries[start_idx:end_idx])
            start_idx = end_idx
        
        print(f"Split {len(entries)} entries into {args.num_gpus} chunks for feature extraction: {[len(chunk) for chunk in entry_chunks]}")
        
        def extract_features_worker(worker_id, device, entries_chunk, images_dir, result_queue, clip_model_name):
            """Worker that extracts features for a chunk of entries."""
            clip_model, clip_processor = load_clip_model(device, clip_model_name)
            results = []
            
            for entry in entries_chunk:
                image_path = entry.get('image', '')
                if not image_path:
                    results.append((entry, None))
                    continue
                
                # Construct full image path
                if Path(image_path).is_absolute():
                    full_image_path = Path(image_path)
                else:
                    full_image_path = Path(image_path)
                    if not full_image_path.exists():
                        image_filename = Path(image_path).name
                        full_image_path = images_dir / image_filename
                
                if not full_image_path.exists():
                    results.append((entry, None))
                    continue
                
                # Extract CLIP features
                features = extract_features_for_image(clip_model, clip_processor, str(full_image_path), device)
                results.append((entry, features))
            
            result_queue.put((worker_id, results))
        
        # Create queue for results
        manager = Manager()
        result_queue = manager.Queue()
        
        # Start worker processes
        processes = []
        for worker_id, (device, chunk) in enumerate(zip(devices, entry_chunks)):
            if len(chunk) > 0:
                p = Process(
                    target=extract_features_worker,
                    args=(worker_id, device, chunk, images_dir, result_queue, args.clip_model)
                )
                p.start()
                processes.append(p)
        
        # Collect results from all workers
        worker_results = {}
        for _ in range(len(processes)):
            worker_id, results = result_queue.get()
            worker_results[worker_id] = results
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Reconstruct entry_features in original order
        entry_features = []
        for worker_id in range(len(entry_chunks)):
            if worker_id in worker_results:
                entry_features.extend(worker_results[worker_id])
        
        print(f"\nExtracted features for {len(entry_features)} entries")
        print(f"Skipped {sum(1 for _, f in entry_features if f is None)} entries (missing images or errors)")
    else:
        # Single GPU: extract features sequentially
        clip_model, clip_processor = load_clip_model(devices[0], args.clip_model)
        entry_features = []
        
        pbar = tqdm(entries, desc="Extracting features", unit="image", total=len(entries))
        for entry in pbar:
            image_path = entry.get('image', '')
            if not image_path:
                entry_features.append((entry, None))
                pbar.set_postfix({'processed': len(entry_features), 'skipped': sum(1 for _, f in entry_features if f is None)})
                continue
            
            # Construct full image path
            if Path(image_path).is_absolute():
                full_image_path = Path(image_path)
            else:
                full_image_path = Path(image_path)
                if not full_image_path.exists():
                    image_filename = Path(image_path).name
                    full_image_path = images_dir / image_filename
            
            if not full_image_path.exists():
                entry_features.append((entry, None))
                pbar.set_postfix({'processed': len(entry_features), 'skipped': sum(1 for _, f in entry_features if f is None)})
                continue
            
            # Extract CLIP features
            features = extract_features_for_image(clip_model, clip_processor, str(full_image_path), devices[0])
            entry_features.append((entry, features))
            pbar.set_postfix({'processed': len(entry_features), 'skipped': sum(1 for _, f in entry_features if f is None)})
        
        print(f"\nExtracted features for {len(entry_features)} entries")
        print(f"Skipped {sum(1 for _, f in entry_features if f is None)} entries (missing images or errors)")
    
    # Step 5b: Assign clusters based on features
    print("\n" + "=" * 80)
    print("Step 5b: Assigning clusters based on extracted features")
    print("=" * 80)
    
    # Create output files
    output_files = {}
    for cluster_id in range(n_clusters):
        output_file = output_dir / f"{output_prefix}_cluster{cluster_id}.jsonl"
        output_files[cluster_id] = open(output_file, 'w')
    
    processed = 0
    skipped = 0
    cluster_counts = {i: 0 for i in range(n_clusters)}
    
    pbar = tqdm(entry_features, desc="Assigning clusters", unit="entry", total=len(entry_features))
    for entry, features in pbar:
        if features is None:
            skipped += 1
            pbar.set_postfix({
                'processed': processed, 
                'skipped': skipped, 
                **{f'cluster{i}': cluster_counts[i] for i in range(n_clusters)}
            })
            continue
        
        # Assign to cluster using the appropriate function
        cluster_id = assign_func(features)
        
        # Write to appropriate cluster file
        output_files[cluster_id].write(json.dumps(entry) + '\n')
        cluster_counts[cluster_id] += 1
        processed += 1
        
        pbar.set_postfix({
            'processed': processed, 
            'skipped': skipped, 
            **{f'cluster{i}': cluster_counts[i] for i in range(n_clusters)}
        })
    
    # Close output files
    for f in output_files.values():
        f.close()
    
    # Print statistics
    print("\n" + "=" * 80)
    print("SPLITTING COMPLETE")
    print("=" * 80)
    print(f"Clustering type: {clustering_type}")
    print(f"Total entries processed: {processed}")
    print(f"Total entries skipped: {skipped}")
    print(f"\nCluster distribution:")
    for cluster_id in range(n_clusters):
        count = cluster_counts[cluster_id]
        percentage = (count / processed * 100) if processed > 0 else 0
        print(f"  Cluster {cluster_id}: {count} entries ({percentage:.1f}%)")
    
    print(f"\nOutput files:")
    for cluster_id in range(n_clusters):
        output_file = output_dir / f"{output_prefix}_cluster{cluster_id}.jsonl"
        print(f"  Cluster {cluster_id}: {output_file}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
