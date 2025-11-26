#!/usr/bin/env python3
"""
Process AOKVQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads AOKVQA aokvqa_v1p0_train.json
2. Groups all questions per image into one multi-turn conversation
3. Formats multiple choice questions with letters (A, B, C, D)
4. Converts correct_choice_idx (0-based) to letter (A, B, C, D)
5. Adds prompt "Answer with the option's letter from the given choices directly." after whitespace
6. Extracts image dimensions
7. Uses relative paths starting from data/
8. Outputs JSONL file compatible with InternVL training
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


def format_answer_choices(choices):
    """Format answer choices as letters (A, B, C, D)."""
    formatted = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)  # A, B, C, D, ...
        formatted.append(f"{letter}. {choice}")
    return formatted


def get_answer_letter(correct_choice_idx):
    """Get the answer as a letter (A, B, C, D)."""
    return chr(ord('A') + correct_choice_idx)


def process_aokvqa(
    json_file,
    images_dir,
    output_file,
    max_images=None
):
    """
    Process AOKVQA dataset into InternVL format.
    
    Args:
        json_file: Path to aokvqa_v1p0_train.json
        images_dir: Directory containing AOKVQA images (train2017)
        output_file: Output JSON file path
        max_images: Maximum number of images to process (None for all)
    """
    
    print("Loading AOKVQA data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} questions")
    
    # Group questions by image_id
    print("Grouping questions by image...")
    image_to_questions = defaultdict(list)
    
    for entry in data:
        image_id = entry.get('image_id')
        if image_id is not None:
            image_to_questions[image_id].append(entry)
    
    print(f"Found {len(image_to_questions):,} unique images")
    print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
    
    # Process each image
    prompt = "Answer with the option's letter from the given choices directly."
    
    image_ids = sorted(image_to_questions.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images...")
    print(f"Saving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    total_qa_pairs = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_id in tqdm(image_ids):
        qa_pairs = image_to_questions[image_id]
        
        # Sort by question_id for consistency
        qa_pairs.sort(key=lambda x: x.get('question_id', ''))
        
        # Build image path (AOKVQA uses COCO format: 000000{image_id:012d}.jpg)
        image_filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
            
            # Get relative path starting from "data/" directory
            if "/data/" in image_path:
                data_index = image_path.find("/data/") + 1  # +1 to include the leading /
                relative_path = image_path[data_index:]
            else:
                # Fallback: construct path assuming standard structure
                # AOKVQA images are in coco/train2017
                relative_path = f"data/coco/train2017/{image_filename}"
        
        # Get image dimensions (use absolute path for reading)
        width, height = get_image_dimensions(image_path)
        
        # Build conversations
        conversations = []
        
        for idx, q_data in enumerate(qa_pairs):
            question = q_data.get('question', '')
            choices = q_data.get('choices', [])
            correct_choice_idx = q_data.get('correct_choice_idx')
            
            # Format answer choices as letters
            formatted_choices = format_answer_choices(choices)
            
            # Get answer letter
            answer = get_answer_letter(correct_choice_idx) if correct_choice_idx is not None else ''
            
                # First question includes image tag and prompt (with whitespace separator)
            if idx == 0:
                    human_value = f"<image>\n{question}\n" + "\n".join(formatted_choices) + f" {prompt}"
            else:
                # Subsequent questions don't include the prompt
                human_value = f"{question}\n" + "\n".join(formatted_choices)
            
            conversations.append({
                "from": "human",
                "value": human_value
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })
                total_qa_pairs += 1
        
        # Create entry
        entry = {
                "id": entry_id,
                "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            entry_id += 1
    
    print(f"\nDone! Processed {entry_id:,} images")
    print(f"Total conversations: {entry_id:,}")
    print(f"Total QA pairs: {total_qa_pairs:,}")


def main():
    parser = argparse.ArgumentParser(description="Process AOKVQA dataset for InternVL training")
    parser.add_argument(
        "--json_file",
        type=str,
        default="/media/data2/maschan/internvl/data/aokvqa/aokvqa_v1p0_train.json",
        help="Path to aokvqa_v1p0_train.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/coco/train2017",
        help="Directory containing AOKVQA images (train2017)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/aokvqa/train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_aokvqa(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

