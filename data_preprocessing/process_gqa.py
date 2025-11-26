#!/usr/bin/env python3
"""
Process GQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads GQA train_balanced_questions.json
2. Groups all questions per image into one multi-turn conversation
3. Adds prompt "Answer the question using a single word or phrase." after first question (with whitespace)
4. Uses questions and answers as-is (no capitalization)
5. Extracts image dimensions
6. Outputs JSONL file with relative paths starting from data/
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


def process_gqa(
    json_file,
    images_dir,
    output_file,
    max_images=None
):
    """
    Process GQA dataset into InternVL format.

    Args:
        json_file: Path to train_balanced_questions.json
        images_dir: Directory containing GQA images
        output_file: Output JSON file path
        max_images: Maximum number of images to process (None for all)
    """
    
    print("Loading GQA data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} questions")
    
    # Group questions by image_id
    print("Grouping questions by image...")
    image_to_questions = defaultdict(list)
    
    for key, entry in data.items():
        image_id = entry.get('imageId')
        if image_id:
            image_to_questions[image_id].append(entry)
    
    print(f"Found {len(image_to_questions):,} unique images")
    print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
    
    # Process each image
    image_ids = sorted(image_to_questions.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images...")
    
    # Save output as JSONL
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    total_qa_pairs = 0
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_id in tqdm(image_ids):
        qa_pairs = image_to_questions[image_id]
        
        # Sort by question_id for consistency
        qa_pairs.sort(key=lambda x: int(x.get('questionId', 0)) if isinstance(x.get('questionId'), (int, str)) and str(x.get('questionId')).isdigit() else 0)
        
            # Build image path
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(images_dir, image_filename)
            
            # Get relative path starting from "data/" directory
            if "/data/" in images_dir:
                data_index = images_dir.find("/data/") + 1  # +1 to include the leading /
                relative_path = images_dir[data_index:] + "/" + image_filename
            else:
                # Fallback: construct path assuming standard structure
                dir_name = os.path.basename(images_dir)
                parent_dir = os.path.basename(os.path.dirname(images_dir))
                relative_path = f"data/{parent_dir}/{dir_name}/{image_filename}"
        
            # Get image dimensions (use absolute path for file access)
        width, height = get_image_dimensions(image_path)
        
        # Build conversations
        conversations = []
        
        for idx, q_data in enumerate(qa_pairs):
                question = q_data.get('question', '')  # Use question as-is (no capitalization)
                answer = q_data.get('answer', '')  # Use answer as-is (no capitalization)
            
                # Prompt (only for first question, with whitespace separator to match evaluation)
                prompt = "Answer the question using a single word or phrase."
            
                # First question includes image tag and prompt (with whitespace, not newline)
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
        
            total_qa_pairs += len(qa_pairs)
            
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
    
    print(f"Done! Processed {entry_id} images")
    print(f"Total conversations: {entry_id}")
    print(f"Total QA pairs: {total_qa_pairs:,}")


def main():
    parser = argparse.ArgumentParser(description="Process GQA dataset for InternVL training")
    parser.add_argument(
        "--json_file",
        type=str,
        default="/media/data2/maschan/internvl/data/gqa/train_balanced_questions.json",
        help="Path to train_balanced_questions.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/gqa/images",
        help="Directory containing GQA images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/gqa/train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_gqa(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()















