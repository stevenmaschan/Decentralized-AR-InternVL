#!/usr/bin/env python3
"""
Preprocess ViQuAE dataset into InternVL-compatible format.

This script:
1. Loads JSONL file with questions and answers
2. Groups entries by image filename
3. Merges all QA pairs for the same image into a single conversation
4. Formats conversations with prompt: "Answer the question using a single word or phrase."
5. Uses only filename (path inside images directory) for image paths
"""

import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from collections import defaultdict


def process_viquae(
    input_file,
    images_dir,
    output_file
):
    """
    Process ViQuAE dataset into InternVL format.
    
    Args:
        input_file: Path to input JSONL file
        images_dir: Directory containing images
        output_file: Output JSON file path
        prompt: Prompt to add to questions
    """
    print("Loading ViQuAE dataset...")
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Loaded {len(entries):,} entries")
    
    print("Grouping entries by image...")
    # Group entries by image filename
    entries_by_image = defaultdict(list)
    for entry in entries:
        image_filename = entry['image']
        entries_by_image[image_filename].append(entry)
    
    print(f"Found {len(entries_by_image):,} unique images")
    
    print("Processing dataset...")
    output_data = []
    missing_images = 0
    
    for image_filename, image_entries in tqdm(entries_by_image.items()):
        # Use only filename (path inside images directory)
        image_path = image_filename
        
        # Check if image exists and get dimensions
        full_image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(full_image_path):
            missing_images += 1
            continue
        
        # Get image dimensions
        try:
            with Image.open(full_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not get dimensions for {image_filename}: {e}")
            width, height = 0, 0
        
        # Build conversations for all QA pairs for this image
        conversations = []
        first_conversation = True
        
        for entry in image_entries:
            question = entry.get('input', '')
            answer_data = entry.get('output', {})
            
            # Get answer - use original_answer if available, otherwise first from answer list
            if isinstance(answer_data, dict):
                answer = answer_data.get('original_answer', '')
                if not answer and 'answer' in answer_data:
                    answer_list = answer_data['answer']
                    if isinstance(answer_list, list) and len(answer_list) > 0:
                        answer = answer_list[0]
            else:
                answer = str(answer_data) if answer_data else ''
            
            # Skip if no question or answer
            if not question or not answer:
                continue
            
            # Add human question (with <image> tag only for first conversation)
            if first_conversation:
                human_value = f"<image>\n{question}"
                first_conversation = False
            else:
                human_value = question
            
            conversations.append({
                "from": "human",
                "value": human_value
            })
            
            # Add GPT answer
            conversations.append({
                "from": "gpt",
                "value": answer
            })
        
        # Skip if no valid conversations were created
        if not conversations:
            continue
        
        # Create conversation entry
        entry = {
            "id": len(output_data),
            "image": image_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
        output_data.append(entry)
    
    print(f"\nProcessed {len(output_data):,} entries")
    print(f"Missing images: {missing_images:,}")
    
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ViQuAE dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file (train.jsonl, validation.jsonl, etc.)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/viquae/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    args = parser.parse_args()
    
    process_viquae(
        input_file=args.input_file,
        images_dir=args.images_dir,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()

