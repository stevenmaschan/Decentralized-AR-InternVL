#!/usr/bin/env python3
"""
Process DocVQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads DocVQA JSON files (train/val/test)
2. Groups all questions per image into one conversation
3. Adds prompt "Answer the question using a single word or phrase." after first question (with whitespace)
4. Uses answers as-is (no normalization)
5. Extracts image dimensions
6. Outputs JSONL file with relative paths starting from data/
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm
import re


def normalize_answer(answer):
    """
    Normalize answer by:
    - Converting to lowercase
    - Stripping whitespace
    - Removing extra spaces
    """
    if not answer:
        return ""
    # Convert to lowercase and strip
    normalized = str(answer).lower().strip()
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def get_most_popular_answer(answers):
    """
    Normalize all answers and return the most popular one.
    If there's a tie, return the first one in the original order.
    """
    if not answers:
        return ""
    
    # Normalize all answers
    normalized_answers = [normalize_answer(ans) for ans in answers]
    
    # Count occurrences
    answer_counts = Counter(normalized_answers)
    
    # Get the most common answer
    most_common = answer_counts.most_common(1)[0][0]
    
    return most_common


def capitalize_first_letter(text):
    """Capitalize the first letter of a string."""
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return 0, 0


def process_docvqa(
    train_json_file,
    val_json_file,
    images_dir,
    output_dir,
    max_images=None,
    process_val=True
):
    """
    Process DocVQA dataset into InternVL format.
    
    Args:
        train_json_file: Path to train_v1.0_withQT.json
        val_json_file: Path to val_v1.0_withQT.json (optional for train processing)
        images_dir: Directory containing images
        output_dir: Directory to save output JSON files
        max_images: Maximum number of images to process (None for all)
        process_val: Whether to process validation set
    """
    
    # Process train set
    print("=" * 60)
    print("Processing TRAIN set...")
    print("=" * 60)
    
    print("Loading train data...")
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    
    train_entries = train_data.get('data', [])
    print(f"Loaded {len(train_entries):,} train entries")
    
    # Group questions by image
    print("Grouping questions by image...")
    image_to_questions = defaultdict(list)
    
    for entry in train_entries:
        # Extract image filename from path like "documents/xnbl0037_1.png"
        image_path = entry.get('image', '')
        if not image_path:
            continue
        # Get just the filename (last part after /)
        image_filename = os.path.basename(image_path)
        image_to_questions[image_filename].append(entry)
    
    print(f"Found {len(image_to_questions):,} unique images")
    print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
    
    # Process each image
    train_output_data = []
    
    image_filenames = sorted(image_to_questions.keys())
    if max_images:
        image_filenames = image_filenames[:max_images]
    
    print(f"\nProcessing {len(image_filenames)} images...")
    
    for image_filename in tqdm(image_filenames):
        questions = image_to_questions[image_filename]
        
        # Sort by questionId for consistency
        questions.sort(key=lambda x: x.get('questionId', 0))
        
        # Build image path
        image_path = os.path.join(images_dir, image_filename)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping...")
            continue
        
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
        
        for idx, q_data in enumerate(questions):
            question = q_data.get('question', '')
            answers = q_data.get('answers', [])
            
            if not question:
                continue
            
            # Use first answer as-is (no normalization)
            answer = answers[0] if answers else ""
            
            if not answer:
                # Skip if no valid answer
                continue
            
            # Use question as-is (no capitalization)
            
            # Add prompt only to the first question (with whitespace separator to match evaluation)
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
        
        # Skip if no valid conversations
        if not conversations:
            continue
        
        # Create entry
        entry = {
            "id": len(train_output_data),
            "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
        train_output_data.append(entry)
    
    # Save train output as JSONL
    train_output_file = os.path.join(output_dir, 'train.jsonl')
    print(f"\nSaving train data to {train_output_file}...")
    os.makedirs(output_dir, exist_ok=True)
    
    entry_id = 0
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for entry in train_output_data:
            entry['id'] = entry_id
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            entry_id += 1
    
    print(f"Done! Processed {len(train_output_data)} images")
    print(f"Total QA pairs: {sum(len(entry['conversations']) // 2 for entry in train_output_data):,}")
    
    # Process validation set if requested
    if process_val and val_json_file and os.path.exists(val_json_file):
        print("\n" + "=" * 60)
        print("Processing VAL set...")
        print("=" * 60)
        
        print("Loading val data...")
        with open(val_json_file, 'r') as f:
            val_data = json.load(f)
        
        val_entries = val_data.get('data', [])
        print(f"Loaded {len(val_entries):,} val entries")
        
        # Group questions by image
        print("Grouping questions by image...")
        image_to_questions = defaultdict(list)
        
        for entry in val_entries:
            image_path = entry.get('image', '')
            if not image_path:
                continue
            image_filename = os.path.basename(image_path)
            image_to_questions[image_filename].append(entry)
        
        print(f"Found {len(image_to_questions):,} unique images")
        print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
        
        # Process each image
        val_output_data = []
        
        image_filenames = sorted(image_to_questions.keys())
        if max_images:
            image_filenames = image_filenames[:max_images]
        
        print(f"\nProcessing {len(image_filenames)} images...")
        
        for image_filename in tqdm(image_filenames):
            questions = image_to_questions[image_filename]
            
            # Sort by questionId for consistency
            questions.sort(key=lambda x: x.get('questionId', 0))
            
            # Build image path
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
            
            for idx, q_data in enumerate(questions):
                question = q_data.get('question', '')
                answers = q_data.get('answers', [])
                
                if not question:
                    continue
                
                # Use first answer as-is (no normalization)
                answer = answers[0] if answers else ""
                
                if not answer:
                    continue
                
                # Use question as-is (no capitalization)
                
                # Add prompt only to the first question (with whitespace separator to match evaluation)
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
            
            # Skip if no valid conversations
            if not conversations:
                continue
            
            # Create entry
            entry = {
                "id": len(val_output_data),
                "image": relative_path,
                "width": width,
                "height": height,
                "conversations": conversations
            }
            
            val_output_data.append(entry)
        
        # Save val output as JSONL
        val_output_file = os.path.join(output_dir, 'val.jsonl')
        print(f"\nSaving val data to {val_output_file}...")
        
        entry_id = 0
        with open(val_output_file, 'w', encoding='utf-8') as f:
            for entry in val_output_data:
                entry['id'] = entry_id
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entry_id += 1
        
        print(f"Done! Processed {len(val_output_data)} images")
        print(f"Total QA pairs: {sum(len(entry['conversations']) // 2 for entry in val_output_data):,}")


def main():
    parser = argparse.ArgumentParser(description="Process DocVQA dataset for InternVL training")
    parser.add_argument(
        "--train_json_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/docvqa/spdocvqa_qas/train_v1.0_withQT.json",
        help="Path to train_v1.0_withQT.json file"
    )
    parser.add_argument(
        "--val_json_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/docvqa/spdocvqa_qas/val_v1.0_withQT.json",
        help="Path to val_v1.0_withQT.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/docvqa/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/docvqa",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip processing validation set"
    )
    
    args = parser.parse_args()
    
    process_docvqa(
        train_json_file=args.train_json_file,
        val_json_file=args.val_json_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        process_val=not args.skip_val
    )


if __name__ == "__main__":
    main()

