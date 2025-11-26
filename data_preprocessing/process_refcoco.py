#!/usr/bin/env python3
"""
Preprocess RefCOCO dataset into InternVL-compatible JSONL format.

This script:
1. Loads instances.json and refs file
2. Uses the evaluation prompt: "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>"
3. Normalizes bounding boxes to [0, 1000] range
4. Outputs bbox coordinates as [[x1, y1, x2, y2]] in GPT response
5. Creates conversations in InternVL format (JSONL)
6. Uses relative paths starting from data/
"""

import json
import pickle
import os
import argparse
from collections import defaultdict
from tqdm import tqdm


def normalize_coordinates(box, image_width, image_height):
    """Normalize bounding box coordinates to [0, 1000] range."""
    x, y, w, h = box  # COCO format: [x, y, width, height]
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box


def convert_bbox_to_xyxy(bbox):
    """Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def process_refcoco(
    instances_file,
    refs_file,
    images_dir,
    output_file,
    dataset_name='refcoco',
    max_samples=None
):
    """
    Process RefCOCO/RefCOCO+/RefCOCOg dataset.
    
    Args:
        instances_file: Path to instances.json
        refs_file: Path to refs pickle file (if None, will auto-detect based on dataset_name)
        images_dir: Directory containing images
        output_file: Output JSONL file path
        dataset_name: Name of dataset (refcoco, refcoco+, refcocog)
        max_samples: Maximum number of samples to process (for testing)
    """
    # Auto-detect refs file if not provided
    if refs_file is None:
        if dataset_name == 'refcoco':
            refs_file = os.path.join(os.path.dirname(instances_file), 'refs(unc).p')
        elif dataset_name == 'refcoco+':
            refs_file = os.path.join(os.path.dirname(instances_file), 'refs(unc).p')
        elif dataset_name == 'refcocog':
            # refcocog has both google and umd, default to google
            refs_file = os.path.join(os.path.dirname(instances_file), 'refs(google).p')
    
    if not os.path.exists(refs_file):
        raise FileNotFoundError(f"Refs file not found: {refs_file}")
    
    print("Loading data...")
    # Load instances.json
    with open(instances_file, 'r') as f:
        instances_data = json.load(f)
    
    # Load refs file
    with open(refs_file, 'rb') as f:
        refs_data = pickle.load(f)
    
    # Create mappings
    images_dict = {img['id']: img for img in instances_data['images']}
    annotations_dict = {ann['id']: ann for ann in instances_data['annotations']}
    categories_dict = {cat['id']: cat['name'] for cat in instances_data['categories']}
    
    # Map refs to annotations by ann_id
    refs_by_ann_id = defaultdict(list)
    for ref in refs_data:
        ann_id = ref['ann_id']
        refs_by_ann_id[ann_id].append(ref)
    
    print(f"Loaded {len(images_dict)} images, {len(annotations_dict)} annotations, {len(refs_data)} refs")
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann_id, annotation in annotations_dict.items():
        image_id = annotation['image_id']
        if image_id in images_dict:
            annotations_by_image[image_id].append(annotation)
    
    # Limit samples if specified
    image_ids = list(annotations_by_image.keys())
    if max_samples:
        image_ids = image_ids[:max_samples]
        print(f"\nProcessing {max_samples} images for verification...")
    else:
        print(f"\nProcessing {len(image_ids)} images...")
    
    # Evaluation prompt (matches evaluate_grounding.py)
    prompt_template = 'Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>'
    
    # Save output as JSONL
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    total_qa_pairs = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_id in tqdm(image_ids, desc="Processing"):
        image_info = images_dict[image_id]
        image_file = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
            # Build absolute image path for file access
            image_path = os.path.join(images_dir, image_file)
            absolute_image_path = os.path.abspath(image_path)
            
            # Get relative path starting from "data/" directory
            if "/data/" in absolute_image_path:
                data_index = absolute_image_path.find("/data/") + 1  # +1 to include the leading /
                relative_path = absolute_image_path[data_index:]
            else:
                # Fallback: construct path assuming standard structure
                # Extract directory structure
        if 'COCO_train2014_' in image_file:
                    relative_path = f"data/coco/train2014/{image_file}"
        elif 'COCO_val2014_' in image_file:
                    relative_path = f"data/coco/val2014/{image_file}"
        else:
                    # Try to infer from images_dir
                    dir_name = os.path.basename(images_dir)
                    parent_dir = os.path.basename(os.path.dirname(images_dir))
                    relative_path = f"data/{parent_dir}/{dir_name}/{image_file}"
        
        # Collect all conversations for this image
        conversations = []
        first_conversation = True
        
        # Process all annotations for this image
        for annotation in annotations_by_image[image_id]:
            ann_id = annotation['id']
            
            # Get refs for this annotation
            refs = refs_by_ann_id.get(ann_id, [])
            if not refs:
                continue
            
            # Get bbox and normalize
            bbox_coco = annotation['bbox']  # [x, y, w, h]
            normalized_bbox = normalize_coordinates(bbox_coco, image_width, image_height)
            
            # For each ref, create a conversation pair for EACH sentence
            for ref in refs:
                sentences = ref.get('sentences', [])
                if not sentences:
                    continue
                
                # Use ALL sentences from this ref
                for sentence in sentences:
                    sentence_text = sentence.get('sent', '').strip()
                    if not sentence_text:
                        continue
                    
                        # Use evaluation prompt format
                        prompt_text = prompt_template.format(sentence_text)
                    
                        if first_conversation:
                            human_value = f"<image>\n{prompt_text}"
                        else:
                            human_value = prompt_text
                        
                        # GPT answer: bbox coordinates in format [[x1, y1, x2, y2]]
                        # Normalized to [0, 1000] range (matching evaluation script expectations)
                        gpt_value = f"[[{normalized_bbox[0]}, {normalized_bbox[1]}, {normalized_bbox[2]}, {normalized_bbox[3]}]]"
                    
                    conversations.append({
                        "from": "human",
                        "value": human_value
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": gpt_value
                    })
                    
                        total_qa_pairs += 1
                    first_conversation = False
        
        # Create entry for this image (with all conversations)
        if conversations:
            entry = {
                    "id": entry_id,
                    "image": relative_path,
                "width": image_width,
                "height": image_height,
                "conversations": conversations
            }
            
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entry_id += 1
    
    print(f"\nDone!")
    print(f"Processed {entry_id:,} entries")
    print(f"Total QA pairs: {total_qa_pairs:,}")


def main():
    parser = argparse.ArgumentParser(description="Process RefCOCO dataset for InternVL training")
    parser.add_argument(
        "--instances_file",
        type=str,
        default=None,
        help="Path to instances.json (if None, will auto-detect based on dataset_name)"
    )
    parser.add_argument(
        "--refs_file",
        type=str,
        default=None,
        help="Path to refs pickle file (if None, will auto-detect based on dataset_name)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/coco/train2014",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file path (if None, will auto-generate based on dataset_name)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="refcoco",
        choices=['refcoco', 'refcoco+', 'refcocog'],
        help="Dataset name"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect instances_file if not provided
    if args.instances_file is None:
        base_dir = "/media/data2/maschan/internvl/data"
        if args.dataset_name == 'refcoco':
            args.instances_file = os.path.join(base_dir, "refcoco", "instances.json")
        elif args.dataset_name == 'refcoco+':
            args.instances_file = os.path.join(base_dir, "refcoco+", "instances.json")
        elif args.dataset_name == 'refcocog':
            args.instances_file = os.path.join(base_dir, "refcocog", "instances.json")
    
    # Auto-detect output_file if not provided
    if args.output_file is None:
        base_dir = "/media/data2/maschan/internvl/data"
        if args.dataset_name == 'refcoco':
            args.output_file = os.path.join(base_dir, "refcoco", "train.jsonl")
        elif args.dataset_name == 'refcoco+':
            args.output_file = os.path.join(base_dir, "refcoco+", "train.jsonl")
        elif args.dataset_name == 'refcocog':
            args.output_file = os.path.join(base_dir, "refcocog", "train.jsonl")
    
    process_refcoco(
        instances_file=args.instances_file,
        refs_file=args.refs_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

