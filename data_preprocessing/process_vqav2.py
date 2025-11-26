#!/usr/bin/env python3
"""
Process VQAv2 dataset into InternVL-compatible JSONL format.

This script:
1. Reads VQAv2 questions and annotations
2. Groups questions by image_id (all QA pairs for an image in one conversation)
3. Formats conversations with prompt (only on first question, with whitespace separator)
4. Keeps answers unchanged from raw annotations
5. Includes image dimensions
6. Outputs JSONL file compatible with InternVL training
"""

import json
import os
import argparse
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return 0, 0


def process_vqav2(
    questions_file,
    annotations_file,
    images_dir,
    output_file,
    max_images=None
):
    """
    Process VQAv2 dataset into InternVL format.
    
    Args:
        questions_file: Path to VQAv2 questions JSON file
        annotations_file: Path to VQAv2 annotations JSON file
        images_dir: Directory containing COCO images
        output_file: Output JSON file path
        max_images: Maximum number of images to process (None for all)
    """
    
    print("Loading questions...")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    # Create mapping: question_id -> annotation
    print("Creating question-annotation mapping...")
    ann_map = {ann['question_id']: ann for ann in annotations_data['annotations']}
    
    # Group questions by image_id
    print("Grouping questions by image...")
    image_questions = defaultdict(list)
    for q in questions_data['questions']:
        image_id = q['image_id']
        question_id = q['question_id']
        if question_id in ann_map:
            image_questions[image_id].append({
                'question_id': question_id,
                'question': q['question'],
                'annotation': ann_map[question_id]
            })
    
    print(f"Found {len(image_questions)} unique images")
    print(f"Total question-answer pairs: {sum(len(qs) for qs in image_questions.values())}")
    
    # Process each image
    prompt = "Answer the question using a single word or phrase."
    
    image_ids = sorted(image_questions.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images...")
    
    # Open output file for writing JSONL
    entry_id = 0
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_id in tqdm(image_ids):
        qa_pairs = image_questions[image_id]
        
        # Sort by question_id for consistency
        qa_pairs.sort(key=lambda x: x['question_id'])
        
            # Build image path
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
            
            # Get relative path starting from "data/" directory
            # Find "data/" in the path and use everything after it
            if "/data/" in images_dir:
                data_index = images_dir.find("/data/") + 1  # +1 to include the leading /
                relative_path = images_dir[data_index:] + "/" + image_filename
            else:
                # Fallback: construct path assuming standard structure
                # Extract directory name (e.g., "train2014") and parent (e.g., "vqav2")
                dir_name = os.path.basename(images_dir)
                parent_dir = os.path.basename(os.path.dirname(images_dir))
                relative_path = f"data/{parent_dir}/{dir_name}/{image_filename}"
        
            # Get image dimensions (use absolute path for file access)
        width, height = get_image_dimensions(image_path)
        
        # Build conversations
        conversations = []
        for idx, qa in enumerate(qa_pairs):
            question = qa['question']
                # Keep answer unchanged from raw annotations
            answer = qa['annotation']['multiple_choice_answer']
            
                # First question includes the prompt (with whitespace, not newline, to match evaluation)
            if idx == 0:
                    human_value = f"<image>\n{question} {prompt}"
            else:
                human_value = question
            
            conversations.append({
                "from": "human",
                "value": human_value
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })
        
        # Create entry
        entry = {
                "id": entry_id,
                "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
            # Write to JSONL file (one JSON object per line)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            entry_id += 1
    
    # Summary
    print(f"\nDone! Processed {entry_id} images")
    print(f"Total conversations: {entry_id}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process VQAv2 dataset for InternVL training")
    parser.add_argument(
        "--questions_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        help="Path to VQAv2 questions JSON file"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/vqav2/v2_mscoco_train2014_annotations.json",
        help="Path to VQAv2 annotations JSON file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/vqav2/train2014",
        help="Directory containing COCO train2014 images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/vqav2/vqav2_train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_vqav2(
        questions_file=args.questions_file,
        annotations_file=args.annotations_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

