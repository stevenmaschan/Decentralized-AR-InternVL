#!/usr/bin/env python3
"""
Unified script to split MME dataset into N clusters for N-expert evaluation.
Supports both two-stage clustering (fine+coarse) and single-stage balanced k-means.

MME structure: category subdirectories, each with txt files (question\tgt) and image files.
Output: cluster0_txt_files through clusterN_txt_files with category txt files in format: image\tquestion\tgt

Usage:
    # Two-stage clustering (2 experts)
    python split_mme_unified.py \
        --mme-dir data/mme \
        --output-dir data/mme_2experts \
        --clustering-results-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \
        --images-dir data/mme \
        --num-experts 2 \
        --device cuda:0

    # Single-stage balanced k-means (4 experts)
    python split_mme_unified.py \
        --mme-dir data/mme \
        --output-dir data/mme_4experts \
        --clustering-results-dir clustering/balanced-kmeans_vit-base-patch-16_4-coarse \
        --images-dir data/mme \
        --num-experts 4 \
        --clustering-type balanced \
        --device cuda:0
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Allow loading truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_clustering_results(clustering_results_dir, clustering_type='auto', prefix='clustering'):
    """
    Load clustering results. Auto-detects type if not specified.
    
    Args:
        clustering_results_dir: Directory containing clustering results
        clustering_type: 'two_stage', 'balanced', or 'auto' (auto-detect)
        prefix: Prefix for clustering result files (for two-stage)
    
    Returns:
        For two-stage: (fine_centroids, coarse_centroids, fine_to_coarse, n_clusters)
        For balanced: (centroids, n_clusters)
    """
    clustering_results_dir = Path(clustering_results_dir)
    
    # Auto-detect clustering type
    if clustering_type == 'auto':
        centroids_file = clustering_results_dir / "clustering_centroids.npy"
        fine_centroids_file = clustering_results_dir / f"{prefix}_fine_centroids.npy"
        
        if centroids_file.exists():
            clustering_type = 'balanced'
            print("Auto-detected: single-stage balanced k-means")
        elif fine_centroids_file.exists():
            clustering_type = 'two_stage'
            print("Auto-detected: two-stage clustering")
        else:
            raise FileNotFoundError(
                f"Could not detect clustering type. Expected either:\n"
                f"  - {centroids_file} (balanced k-means)\n"
                f"  - {fine_centroids_file} (two-stage)"
            )
    
    if clustering_type == 'balanced':
        # Single-stage balanced k-means
        centroids_file = clustering_results_dir / "clustering_centroids.npy"
        if not centroids_file.exists():
            raise FileNotFoundError(f"Centroids not found: {centroids_file}")
        
        centroids = np.load(centroids_file)
        n_clusters = len(centroids)
        print(f"Loaded centroids: shape {centroids.shape}")
        print(f"Number of clusters: {n_clusters}")
        return ('balanced', centroids, n_clusters)
    
    elif clustering_type == 'two_stage':
        # Two-stage clustering
        fine_centroids_file = clustering_results_dir / f"{prefix}_fine_centroids.npy"
        coarse_centroids_file = clustering_results_dir / f"{prefix}_coarse_centroids.npy"
        fine_to_coarse_file = clustering_results_dir / f"{prefix}_fine_to_coarse_mapping.npy"
        
        if not fine_centroids_file.exists():
            raise FileNotFoundError(f"Fine centroids not found: {fine_centroids_file}")
        if not coarse_centroids_file.exists():
            raise FileNotFoundError(f"Coarse centroids not found: {coarse_centroids_file}")
        if not fine_to_coarse_file.exists():
            raise FileNotFoundError(f"Fine-to-coarse mapping not found: {fine_to_coarse_file}")
        
        fine_centroids = np.load(fine_centroids_file)
        coarse_centroids = np.load(coarse_centroids_file)
        fine_to_coarse = np.load(fine_to_coarse_file)
        n_clusters = len(coarse_centroids)
        
        print(f"Loaded fine centroids: shape {fine_centroids.shape}")
        print(f"Loaded coarse centroids: shape {coarse_centroids.shape}")
        print(f"Loaded fine-to-coarse mapping: shape {fine_to_coarse.shape}")
        print(f"Number of clusters: {n_clusters}")
        return ('two_stage', fine_centroids, coarse_centroids, fine_to_coarse, n_clusters)
    
    else:
        raise ValueError(f"Unknown clustering type: {clustering_type}")


def extract_clip_features_parallel(images_dir, image_paths, device, num_gpus=1, clip_model="openai/clip-vit-base-patch16"):
    """Extract CLIP features for images in parallel across GPUs."""
    print(f"Loading CLIP model: {clip_model}...")
    
    # Load models on each GPU
    models = {}
    processors = {}
    for gpu_id in range(num_gpus):
        gpu_device = f'cuda:{gpu_id}'
        processors[gpu_id] = CLIPProcessor.from_pretrained(clip_model)
        models[gpu_id] = CLIPModel.from_pretrained(clip_model).to(gpu_device)
        models[gpu_id].eval()
    
    # Split work across GPUs with proper remainder handling
    chunk_size = len(image_paths) // num_gpus
    remainder = len(image_paths) % num_gpus
    
    all_features = [None] * len(image_paths)
    image_to_idx = {img: idx for idx, img in enumerate(image_paths)}
    
    def process_chunk(chunk_data):
        gpu_id, chunk_images = chunk_data
        gpu_device = f'cuda:{gpu_id}'
        processor = processors[gpu_id]
        model = models[gpu_id]
        
        chunk_features = []
        for img_path in chunk_images:
            full_path = Path(images_dir) / img_path
            if not full_path.exists():
                chunk_features.append(None)
                continue
            
            try:
                image = Image.open(full_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt").to(gpu_device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    chunk_features.append(image_features.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
                chunk_features.append(None)
        
        return chunk_images, chunk_features
    
    # Process chunks
    chunks = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        current_chunk_size = chunk_size + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunk_images = image_paths[start_idx:end_idx]
        if chunk_images:
            chunks.append((gpu_id, chunk_images))
        start_idx = end_idx
    
    print(f"Processing {len(image_paths)} images across {num_gpus} GPUs...")
    for gpu_id, chunk_images in tqdm(chunks, desc="Extracting features"):
        chunk_images_list, chunk_features = process_chunk((gpu_id, chunk_images))
        for img, feat in zip(chunk_images_list, chunk_features):
            idx = image_to_idx[img]
            all_features[idx] = feat
    
    return all_features, image_to_idx


def assign_to_cluster_two_stage(features, fine_centroids, fine_to_coarse):
    """Assign features to clusters using two-stage clustering."""
    # Normalize features
    features_norm = features / np.linalg.norm(features)
    fine_centroids_norm = fine_centroids / np.linalg.norm(fine_centroids, axis=1, keepdims=True)
    
    # Assign to fine cluster
    similarities = np.dot(fine_centroids_norm, features_norm)
    fine_cluster_id = np.argmax(similarities)
    
    # Map to coarse cluster
    coarse_cluster_id = fine_to_coarse[fine_cluster_id]
    
    return coarse_cluster_id


def assign_to_cluster_balanced(features, centroids):
    """Assign features to clusters using single-stage balanced k-means."""
    # Normalize features
    features_norm = features / np.linalg.norm(features)
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Assign to closest centroid
    similarities = np.dot(centroids_norm, features_norm)
    cluster_id = np.argmax(similarities)
    
    return cluster_id


def split_mme_txt_files(mme_dir, output_dir, clustering_results_dir, images_dir, 
                        num_experts, device='cuda:0', num_gpus=1, 
                        clustering_type='auto', prefix='clustering', 
                        clip_model='openai/clip-vit-base-patch16'):
    """
    Split MME txt files into N clusters for N-expert evaluation.
    
    Args:
        mme_dir: MME directory containing category subdirectories
        output_dir: Output directory for cluster txt files
        clustering_results_dir: Directory with clustering results
        images_dir: Base directory for MME images
        num_experts: Number of experts/clusters (2, 4, etc.)
        device: Base device to use
        num_gpus: Number of GPUs for feature extraction
        clustering_type: 'two_stage', 'balanced', or 'auto'
        prefix: Prefix for clustering result files (two-stage only)
        clip_model: CLIP model to use
    """
    mme_dir = Path(mme_dir)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)
    
    # Create output directories
    cluster_dirs = {}
    for i in range(num_experts):
        cluster_dirs[i] = output_dir / f'cluster{i}_txt_files'
        cluster_dirs[i].mkdir(parents=True, exist_ok=True)
    
    # Load clustering results
    print("\n" + "=" * 80)
    print("Loading clustering results")
    print("=" * 80)
    clustering_data = load_clustering_results(clustering_results_dir, clustering_type, prefix)
    
    clustering_type_actual = clustering_data[0]
    n_clusters = clustering_data[-1]  # Last element is always n_clusters
    
    # Verify number of clusters matches num_experts
    if n_clusters != num_experts:
        print(f"Warning: Number of clusters ({n_clusters}) does not match --num-experts ({num_experts})")
        print(f"Using {n_clusters} clusters from clustering results")
        num_experts = n_clusters
        # Update cluster_dirs if needed
        for i in range(num_experts):
            if i not in cluster_dirs:
                cluster_dirs[i] = output_dir / f'cluster{i}_txt_files'
                cluster_dirs[i].mkdir(parents=True, exist_ok=True)
    
    # Collect all entries from all category subdirectories
    all_entries = []  # List of (category, image_name, question, gt, image_path)
    image_to_entries = defaultdict(list)  # image_path -> list of (category, image_name, question, gt)
    
    print("\n" + "=" * 80)
    print("Collecting MME entries")
    print("=" * 80)
    for category_dir in sorted(mme_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        txt_files = sorted(category_dir.glob('*.txt'))
        
        for txt_file in txt_files:
            # Get corresponding image file (try different extensions)
            img_stem = txt_file.stem
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = category_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file is None:
                print(f"Warning: No image found for {txt_file}")
                continue
            
            img_name = img_file.name
            img_path = f"{category_name}/{img_name}"
            
            # Read questions from txt file
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Format: question\tgt
                    parts = line.split('\t')
                    if len(parts) == 2:
                        question, gt = parts
                        all_entries.append((category_name, img_name, question, gt, img_path))
                        image_to_entries[img_path].append((category_name, img_name, question, gt))
    
    print(f"Collected {len(all_entries)} entries from {len(image_to_entries)} unique images")
    
    # Get unique image paths
    unique_images = list(image_to_entries.keys())
    
    # Extract CLIP features for all unique images
    print("\n" + "=" * 80)
    print("Extracting CLIP features")
    print("=" * 80)
    features_list, image_to_idx = extract_clip_features_parallel(
        images_dir, unique_images, device, num_gpus, clip_model
    )
    
    # Filter to only images with features
    valid_images = []
    valid_features = []
    for img in unique_images:
        idx = image_to_idx[img]
        if features_list[idx] is not None:
            valid_images.append(img)
            valid_features.append(features_list[idx])
    
    print(f"Extracted features for {len(valid_images)} images")
    
    # Assign to clusters
    print("\n" + "=" * 80)
    print("Assigning to clusters")
    print("=" * 80)
    if clustering_type_actual == 'two_stage':
        fine_centroids = clustering_data[1]
        coarse_centroids = clustering_data[2]
        fine_to_coarse = clustering_data[3]
        cluster_assignments = [
            assign_to_cluster_two_stage(feat, fine_centroids, fine_to_coarse)
            for feat in tqdm(valid_features, desc="Assigning clusters")
        ]
    else:  # balanced
        centroids = clustering_data[1]
        cluster_assignments = [
            assign_to_cluster_balanced(feat, centroids)
            for feat in tqdm(valid_features, desc="Assigning clusters")
        ]
    
    # Build image to cluster mapping
    image_to_cluster = {}
    for img, cluster_id in zip(valid_images, cluster_assignments):
        image_to_cluster[img] = int(cluster_id)
    
    # Distribute entries to cluster files
    cluster_files = {
        i: defaultdict(list) for i in range(num_experts)  # category -> list of lines
    }
    
    cluster_counts = {i: 0 for i in range(num_experts)}
    
    print("\n" + "=" * 80)
    print("Distributing entries to clusters")
    print("=" * 80)
    for category_name, img_name, question, gt, img_path in tqdm(all_entries, desc="Distributing"):
        cluster_id = image_to_cluster.get(img_path, 0)  # Default to cluster 0 if not found
        if cluster_id >= num_experts:
            cluster_id = cluster_id % num_experts  # Wrap around if needed
        line = f"{img_name}\t{question}\t{gt}\n"
        cluster_files[cluster_id][category_name].append(line)
        cluster_counts[cluster_id] += 1
    
    # Write cluster files
    print("\n" + "=" * 80)
    print("Writing cluster files")
    print("=" * 80)
    for cluster_id in range(num_experts):
        cluster_dir = cluster_dirs[cluster_id]
        for category_name, lines in cluster_files[cluster_id].items():
            output_file = cluster_dir / f"{category_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"  Cluster {cluster_id} - {category_name}: {len(lines)} entries")
    
    total_entries = sum(cluster_counts.values())
    print(f"\n" + "=" * 80)
    print("Cluster distribution")
    print("=" * 80)
    for cluster_id in range(num_experts):
        print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} entries ({100*cluster_counts[cluster_id]/total_entries:.1f}%)")
    
    return cluster_counts


def main():
    parser = argparse.ArgumentParser(
        description='Split MME dataset into N clusters for N-expert evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-stage clustering (2 experts)
  python split_mme_unified.py \\
    --mme-dir data/mme \\
    --output-dir data/mme_2experts \\
    --clustering-results-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \\
    --images-dir data/mme \\
    --num-experts 2 \\
    --device cuda:0

  # Single-stage balanced k-means (4 experts)
  python split_mme_unified.py \\
    --mme-dir data/mme \\
    --output-dir data/mme_4experts \\
    --clustering-results-dir clustering/balanced-kmeans_vit-base-patch-16_4-coarse \\
    --images-dir data/mme \\
    --num-experts 4 \\
    --clustering-type balanced \\
    --device cuda:0
        """
    )
    parser.add_argument('--mme-dir', type=str, required=True, 
                       help='MME directory containing category subdirectories')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for cluster txt files')
    parser.add_argument('--clustering-results-dir', type=str, required=True, 
                       help='Directory with clustering results')
    parser.add_argument('--images-dir', type=str, required=True, 
                       help='Base directory for MME images (same as mme-dir)')
    parser.add_argument('--num-experts', type=int, required=True, 
                       help='Number of experts/clusters (2, 4, etc.)')
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Base device to use (e.g., cuda:0)')
    parser.add_argument('--num-gpus', type=int, default=1, 
                       help='Number of GPUs to use for feature extraction')
    parser.add_argument('--clustering-type', type=str, default='auto', 
                       choices=['auto', 'two_stage', 'balanced'],
                       help='Clustering type: auto-detect, two_stage, or balanced')
    parser.add_argument('--prefix', type=str, default='clustering', 
                       help='Prefix for clustering result files (two-stage only)')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch16', 
                       help='CLIP model to use for feature extraction')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"SPLITTING MME DATASET INTO {args.num_experts} CLUSTERS")
    print("=" * 80)
    print(f"MME directory: {args.mme_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Clustering results: {args.clustering_results_dir}")
    print(f"Images directory: {args.images_dir}")
    print(f"Number of experts: {args.num_experts}")
    print(f"Clustering type: {args.clustering_type}")
    print("=" * 80)
    
    split_mme_txt_files(
        args.mme_dir,
        args.output_dir,
        args.clustering_results_dir,
        args.images_dir,
        args.num_experts,
        args.device,
        args.num_gpus,
        args.clustering_type,
        args.prefix,
        args.clip_model
    )


if __name__ == '__main__':
    main()
