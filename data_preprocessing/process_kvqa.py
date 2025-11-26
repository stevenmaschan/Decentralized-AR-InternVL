#!/usr/bin/env python3
"""
Preprocess KVQA dataset into InternVL-compatible JSONL format.

This script:
1. Loads dataset.json file
2. Groups all questions for the same image into a single conversation entry
3. Adds prompt "Answer the question using a single word or phrase." only to the first question (with whitespace separator)
4. Uses questions and answers as-is (no capitalization)
5. Outputs JSONL format with relative paths starting from data/
"""

import json
import os
import argparse
from tqdm import tqdm
from PIL import Image


def process_kvqa(
    dataset_file,
    images_dir,
    output_file
):
    """
    Process KVQA dataset into InternVL format.
    
    Args:
        dataset_file: Path to dataset.json file
        images_dir: Directory containing KVQAimgs images
        output_file: Output JSON file path
    """
    print("Loading dataset...")
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} entries")
    
    print("Processing dataset...")
    missing_images = 0
    invalid_entries = 0
    
    # Save output as JSONL
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    total_qa_pairs = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for data_entry_id, entry in tqdm(data.items(), desc="Processing entries"):
        # Get image path
        img_path = entry.get('imgPath', '')
        if not img_path:
            invalid_entries += 1
            continue
        
        # Extract just the filename (e.g., "KVQAimgs/21717.jpg" -> "21717.jpg")
        image_filename = os.path.basename(img_path)
        
        # Check if image exists and get dimensions
        full_image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(full_image_path):
            missing_images += 1
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
        try:
            with Image.open(full_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not get dimensions for {image_filename}: {e}")
            width, height = 0, 0
        
            # Get questions and answers (use as-is, no capitalization)
        questions = entry.get('Questions', [])
        answers = entry.get('Answers', [])
        
        # Ensure questions and answers have the same length
        if len(questions) != len(answers):
                print(f"Warning: Mismatch in questions/answers for {data_entry_id}: {len(questions)} questions, {len(answers)} answers")
            min_len = min(len(questions), len(answers))
            questions = questions[:min_len]
            answers = answers[:min_len]
        
        if not questions:
            invalid_entries += 1
            continue
        
        # Build conversations for all QA pairs for this image
        conversations = []
        first_conversation = True
            
            # Prompt (only for first question, with whitespace separator to match evaluation)
            prompt = "Answer the question using a single word or phrase."
        
        for question, answer in zip(questions, answers):
            # Skip if question or answer is empty
            if not question or not answer:
                continue
            
                # Use question and answer as-is (no capitalization)
                question_text = str(question).strip()
                answer_text = str(answer).strip()
                
                # Add human question (with <image> tag and prompt only for first conversation)
            if first_conversation:
                    human_value = f"<image>\n{question_text} {prompt}"  # Whitespace separator
                first_conversation = False
            else:
                    human_value = question_text
            
            conversations.append({
                "from": "human",
                "value": human_value
            })
            
                # Add GPT answer (as-is, no capitalization)
            conversations.append({
                "from": "gpt",
                    "value": answer_text
            })
        
        # Skip if no valid conversations were created
        if not conversations:
            invalid_entries += 1
            continue
        
            total_qa_pairs += len(conversations) // 2
            
        # Create conversation entry
        entry_output = {
                "id": entry_id,
                "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
            f.write(json.dumps(entry_output, ensure_ascii=False) + '\n')
            entry_id += 1
    
    print(f"\nProcessed {entry_id:,} entries")
    print(f"Missing images: {missing_images:,}")
    print(f"Invalid entries: {invalid_entries:,}")
    print(f"Total QA pairs: {total_qa_pairs:,}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess KVQA dataset")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="/media/data2/maschan/internvl/data/kvqa/raw/dataset.json",
        help="Path to KVQA dataset.json file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/kvqa/raw/KVQAimgs",
        help="Directory containing KVQAimgs images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/kvqa/train.jsonl",
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    process_kvqa(
        dataset_file=args.dataset_file,
        images_dir=args.images_dir,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()


