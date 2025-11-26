#!/usr/bin/env python3
"""
Extract CLIP features for updated unique images from individual NPZ files and stack them.
"""

import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def normalize_path_for_matching(path, dataset_name):
    """Normalize image path to match NPZ file keys"""
    # Extract filename or relative path
    if dataset_name == 'train2014':
        # vqav2 uses: COCO_train2014_000000000009.jpg
        filename = os.path.basename(path)
        return filename
    elif dataset_name == 'aokvqa':
        # aokvqa uses: train2017/000000000074.jpg
        if 'train2017' in path:
            filename = os.path.basename(path)
            return f"train2017/{filename}"
        return os.path.basename(path)
    elif dataset_name == 'textvqa':
        # textcaps uses: ../train_images/0000599864fd15b3.jpg
        # TextVQA and TextCaps share images, so use textcaps features
        filename = os.path.basename(path)
        return f"../train_images/{filename}"
    elif dataset_name == 'textcaps':
        # textcaps uses: ../train_images/0000599864fd15b3.jpg
        filename = os.path.basename(path)
        return f"../train_images/{filename}"
    else:
        # For other datasets, try different formats
        filename = os.path.basename(path)
        # Try with relative path
        if '/' in path:
            parts = path.split('/')
            if len(parts) >= 2:
                return f"{parts[-2]}/{filename}"
        return filename

def main():
    print("="*80)
    print("Extracting CLIP Features for Updated Unique Images")
    print("="*80)
    
    # Load updated unique image paths
    updated_file = "/media/data2/maschan/internvl/data/unique_image_paths_updated.txt"
    print(f"\nLoading updated unique image paths from: {updated_file}")
    
    unique_paths = []
    with open(updated_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                unique_paths.append(path)
    
    print(f"Total unique images: {len(unique_paths)}")
    
    # Map paths to datasets
    print("\nMapping images to datasets...")
    dataset_mapping = defaultdict(list)
    
    for i, path in enumerate(unique_paths):
        path_lower = path.lower()
        if 'train2014' in path:
            dataset_mapping['train2014'].append((i, path))
        elif 'train2017' in path:
            dataset_mapping['aokvqa'].append((i, path))
        elif 'textvqa' in path_lower:
            dataset_mapping['textvqa'].append((i, path))
        elif 'textcaps' in path_lower:
            dataset_mapping['textcaps'].append((i, path))
        elif 'ai2d' in path_lower or 'ai2diagram' in path_lower:
            dataset_mapping['ai2d'].append((i, path))
        elif 'chartqa' in path_lower:
            dataset_mapping['chartqa'].append((i, path))
        elif 'docvqa' in path_lower:
            dataset_mapping['docvqa'].append((i, path))
        elif 'gqa' in path_lower:
            dataset_mapping['gqa'].append((i, path))
        elif 'infographicsvqa' in path_lower or 'infovqa' in path_lower:
            dataset_mapping['infographicsvqa'].append((i, path))
        elif 'kvqa' in path_lower:
            dataset_mapping['kvqa'].append((i, path))
        elif 'scienceqa' in path_lower:
            dataset_mapping['scienceqa_image'].append((i, path))
        elif 'sharegpt4o' in path_lower:
            dataset_mapping['sharegpt4o'].append((i, path))
        elif 'allava' in path_lower:
            dataset_mapping['allava_instruct_sampled'].append((i, path))
    
    print(f"\nDataset distribution:")
    for dataset, items in sorted(dataset_mapping.items()):
        print(f"  {dataset}: {len(items)} images")
    
    # CLIP feature file mapping
    clip_feature_files = {
        'train2014': '/home/maschan/ddfm/InternVL/clustering/vqav2/vqav2_clip_features.npz',
        'aokvqa': '/home/maschan/ddfm/InternVL/clustering/aokvqa/aokvqa_clip_features.npz',
        'textvqa': '/home/maschan/ddfm/InternVL/clustering/textcaps/textcaps_clip_features.npz',  # TextVQA and TextCaps share images
        'textcaps': '/home/maschan/ddfm/InternVL/clustering/textcaps/textcaps_clip_features.npz',
        'ai2d': '/home/maschan/ddfm/InternVL/clustering/ai2d/ai2d_clip_features.npz',
        'chartqa': '/home/maschan/ddfm/InternVL/clustering/chartqa/chartqa_clip_features.npz',
        'docvqa': '/home/maschan/ddfm/InternVL/clustering/docvqa/docvqa_clip_features.npz',
        'gqa': '/home/maschan/ddfm/InternVL/clustering/gqa/gqa_clip_features.npz',
        'infographicsvqa': '/home/maschan/ddfm/InternVL/clustering/infovqa/infovqa_clip_features.npz',
        'kvqa': '/home/maschan/ddfm/InternVL/clustering/kvqa/kvqa_clip_features.npz',
        'scienceqa_image': '/home/maschan/ddfm/InternVL/clustering/scienceqa_image/scienceqa_image_clip_features.npz',
        'sharegpt4o': '/home/maschan/ddfm/InternVL/clustering/sharegpt4o/sharegpt4o_clip_features.npz',
        'allava_instruct_sampled': '/home/maschan/ddfm/InternVL/clustering/allava/allava_clip_features.npz',
    }
    
    # Load all NPZ files
    print("\nLoading CLIP feature files...")
    npz_data = {}
    for dataset, filepath in clip_feature_files.items():
        if os.path.exists(filepath):
            print(f"  Loading {dataset}...")
            npz_data[dataset] = np.load(filepath, allow_pickle=True)
            keys = list(npz_data[dataset].keys())
            print(f"    {len(keys)} features loaded")
        else:
            print(f"  WARNING: {dataset} file not found: {filepath}")
    
    # Extract features in order
    print("\nExtracting features...")
    all_features = []
    missing_count = 0
    missing_paths = []
    
    for i, path in enumerate(tqdm(unique_paths, desc="Extracting")):
        # Determine dataset
        path_lower = path.lower()
        if 'train2014' in path:
            dataset = 'train2014'
        elif 'train2017' in path:
            dataset = 'aokvqa'
        elif 'textvqa' in path_lower:
            dataset = 'textvqa'
        elif 'textcaps' in path_lower:
            dataset = 'textcaps'
        elif 'ai2d' in path_lower or 'ai2diagram' in path_lower:
            dataset = 'ai2d'
        elif 'chartqa' in path_lower:
            dataset = 'chartqa'
        elif 'docvqa' in path_lower:
            dataset = 'docvqa'
        elif 'gqa' in path_lower:
            dataset = 'gqa'
        elif 'infographicsvqa' in path_lower or 'infovqa' in path_lower:
            dataset = 'infographicsvqa'
        elif 'kvqa' in path_lower:
            dataset = 'kvqa'
        elif 'scienceqa' in path_lower:
            dataset = 'scienceqa_image'
        elif 'sharegpt4o' in path_lower:
            dataset = 'sharegpt4o'
        elif 'allava' in path_lower:
            dataset = 'allava_instruct_sampled'
        else:
            missing_count += 1
            missing_paths.append(path)
            all_features.append(None)
            continue
        
        if dataset not in npz_data:
            missing_count += 1
            missing_paths.append(path)
            all_features.append(None)
            continue
        
        # Normalize path for matching
        normalized_key = normalize_path_for_matching(path, dataset)
        
        # Try to find the key in NPZ file
        found = False
        npz_keys = list(npz_data[dataset].keys())
        
        # Try exact match first
        if normalized_key in npz_keys:
            feature = npz_data[dataset][normalized_key]
            if feature.shape == (1, 768):
                all_features.append(feature.squeeze(0))
                found = True
        else:
            # Try alternative matches
            filename = os.path.basename(path)
            for key in npz_keys:
                if filename in key or key.endswith(filename):
                    feature = npz_data[dataset][key]
                    if feature.shape == (1, 768):
                        all_features.append(feature.squeeze(0))
                        found = True
                        break
        
        if not found:
            missing_count += 1
            missing_paths.append(path)
            all_features.append(None)
    
    print(f"\nExtracted {len(all_features) - missing_count} features")
    print(f"Missing features: {missing_count}")
    
    if missing_count > 0:
        print(f"\nSample missing paths:")
        for path in missing_paths[:10]:
            print(f"  {path}")
    
    # Filter out None values and stack
    print("\nStacking features...")
    valid_features = []
    valid_indices = []
    for i, feat in enumerate(all_features):
        if feat is not None:
            valid_features.append(feat)
            valid_indices.append(i)
    
    if len(valid_features) == 0:
        print("ERROR: No valid features found!")
        return
    
    stacked_features = np.stack(valid_features, axis=0)
    print(f"Stacked features shape: {stacked_features.shape}")
    
    # Save results
    output_dir = "/home/maschan/ddfm/InternVL/clustering"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "consolidated_clip_features_updated.npy")
    np.save(output_file, stacked_features)
    print(f"\nSaved stacked features to: {output_file}")
    
    # Save metadata
    metadata = {
        'total_images': len(unique_paths),
        'valid_features': len(valid_features),
        'missing_features': missing_count,
        'feature_shape': stacked_features.shape,
        'valid_indices': valid_indices[:100] if len(valid_indices) > 100 else valid_indices,  # Sample
        'missing_paths': missing_paths[:100] if len(missing_paths) > 100 else missing_paths,  # Sample
    }
    
    metadata_file = os.path.join(output_dir, "consolidated_clip_features_updated_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Total images: {len(unique_paths)}")
    print(f"Valid features: {len(valid_features)}")
    print(f"Missing features: {missing_count}")
    print(f"Stacked features shape: {stacked_features.shape}")

if __name__ == "__main__":
    main()



