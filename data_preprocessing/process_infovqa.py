#!/usr/bin/env python3
"""
Process InfoVQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads infographicsVQA_train_v1.0.json
2. Groups all questions per image into one conversation
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


def process_infovqa(
    json_file,
    images_dir,
    output_file,
    max_images=None
):
    """
    Process InfoVQA dataset into InternVL format.
    
    Args:
        json_file: Path to infographicsVQA_train_v1.0.json
        images_dir: Directory containing images
        output_file: Output JSON file path
        max_images: Maximum number of images to process (None for all)
    """
    
    print("Loading InfoVQA data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    entries = data['data']
    print(f"Loaded {len(entries):,} entries")
    
    # Group questions by image
    print("Grouping questions by image...")
    image_to_questions = defaultdict(list)
    
    for entry in entries:
        image_name = entry['image_local_name']
        image_to_questions[image_name].append(entry)
    
    print(f"Found {len(image_to_questions):,} unique images")
    print(f"Total questions: {sum(len(qs) for qs in image_to_questions.values()):,}")
    
    # Process each image
    image_names = sorted(image_to_questions.keys())
    if max_images:
        image_names = image_names[:max_images]
    
    print(f"\nProcessing {len(image_names)} images...")
    
    output_data = []
    for image_name in tqdm(image_names):
        questions = image_to_questions[image_name]
        
        # Build image path
        image_path = os.path.join(images_dir, image_name)
        
        # Get relative path starting from "data/" directory
        if "/data/" in images_dir:
            data_index = images_dir.find("/data/") + 1  # +1 to include the leading /
            relative_path = images_dir[data_index:] + "/" + image_name
        else:
            # Fallback: construct path assuming standard structure
            dir_name = os.path.basename(images_dir)
            parent_dir = os.path.basename(os.path.dirname(images_dir))
            relative_path = f"data/{parent_dir}/{dir_name}/{image_name}"
        
        # Get image dimensions (use absolute path for file access)
        width, height = get_image_dimensions(image_path)
        
        # Build conversations
        conversations = []
        
        for idx, q_data in enumerate(questions):
            question = q_data['question']  # Use question as-is (no capitalization)
            # Use first answer from answers list as-is (no capitalization)
            answer = q_data['answers'][0] if q_data['answers'] else ""
            
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
        
        # Create entry
        entry = {
            "id": len(output_data),
            "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
        output_data.append(entry)
    
    # Save output as JSONL
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            entry['id'] = entry_id
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            entry_id += 1
    
    print(f"Done! Processed {len(output_data)} images")
    print(f"Total conversations: {len(output_data)}")
    print(f"Total QA pairs: {sum(len(entry['conversations']) // 2 for entry in output_data)}")


def main():
    parser = argparse.ArgumentParser(description="Process InfoVQA dataset for InternVL training")
    parser.add_argument(
        "--json_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/infovqa/infographicsvqa_qas/infographicsVQA_train_v1.0.json",
        help="Path to infographicsVQA_train_v1.0.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/infovqa/infographicsvqa_images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/infographicsvqa/train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_infovqa(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()


