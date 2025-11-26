#!/usr/bin/env python3
"""
Process TextVQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads TextVQA_0.5.1_train.json
2. Groups all questions per image into one conversation
3. Normalizes answers and picks the most popular one
4. Uses questions as-is (no capitalization)
5. Adds prompt after first question (with whitespace separator)
6. Extracts image dimensions
7. Outputs JSONL file with relative paths starting from data/
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
    normalized = answer.lower().strip()
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


def process_textvqa(
    json_file,
    images_dir,
    output_file,
    max_images=None
):
    """
    Process TextVQA dataset into InternVL format.
    
    Args:
        json_file: Path to TextVQA_0.5.1_train.json
        images_dir: Directory containing images
        output_file: Output JSON file path
        max_images: Maximum number of images to process (None for all)
    """
    
    print("Loading TextVQA data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    entries = data['data']
    print(f"Loaded {len(entries):,} entries")
    
    # Filter only train entries
    train_entries = [e for e in entries if e.get('set_name') == 'train']
    print(f"Train entries: {len(train_entries):,}")
    
    # Group questions by image_id
    print("Grouping questions by image...")
    image_to_questions = defaultdict(list)
    
    for entry in train_entries:
        image_id = entry['image_id']
        image_to_questions[image_id].append(entry)
    
    print(f"Found {len(image_to_questions):,} unique images")
    print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
    
    # Process each image
    image_ids = sorted(image_to_questions.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images...")
    
    # Save output as JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_id in tqdm(image_ids):
        questions = image_to_questions[image_id]
        
        # Sort by question_id for consistency
        questions.sort(key=lambda x: x['question_id'])
        
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
        
        for idx, q_data in enumerate(questions):
                question = q_data['question']  # Use question as-is
            answers = q_data.get('answers', [])
            
            # Normalize and get most popular answer
            answer = get_most_popular_answer(answers)
            
                # Prompt (only for first question, with whitespace separator to match evaluation)
                prompt = "Answer the question using a single word or phrase."
            
                # First question includes image tag and prompt
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
    
    print(f"\nDone! Processed {entry_id} images")
    print(f"Total conversations: {entry_id}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process TextVQA dataset for InternVL training")
    parser.add_argument(
        "--json_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/textvqa/TextVQA_0.5.1_train.json",
        help="Path to TextVQA_0.5.1_train.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/textvqa/train_images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/textvqa/textvqa_train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_textvqa(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

