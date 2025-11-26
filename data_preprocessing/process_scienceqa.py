#!/usr/bin/env python3
"""
Process ScienceQA dataset into InternVL-compatible JSONL format.

This script:
1. Reads ScienceQA problems.json and pid_splits.json
2. Processes train split questions
3. For questions with images: Groups all questions per image into one conversation
4. For questions without images: Uses standard InternVL chat format (no image field)
5. Formats multiple choice questions with letters (A, B, C, D, E) to match evaluation
6. Includes hints before questions (if present) to match evaluation format
7. Uses prompt "Answer with the option's letter from the given choices directly."
8. Separates prompt with '\n' to match evaluation format
9. Extracts image dimensions for questions with images
10. Outputs JSONL file with relative paths starting from data/
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
    """Format answer choices as letters (A, B, C, D, E) to match evaluation format."""
    formatted = []
    multiple_choices = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        formatted.append(f"{multiple_choices[i]}. {choice}")
    return formatted


def get_answer_letter(correct_answer_idx):
    """Get the answer as a letter (A, B, C, D, E) to match evaluation format."""
    multiple_choices = ['A', 'B', 'C', 'D', 'E']
    if 0 <= correct_answer_idx < len(multiple_choices):
        return multiple_choices[correct_answer_idx]
    return ""


def process_scienceqa(
    problems_file,
    splits_file,
    base_dir,
    split_name,
    output_file_images,
    output_file_text,
    max_problems=None
):
    """
    Process ScienceQA dataset into InternVL format.
    
    Args:
        problems_file: Path to problems.json
        splits_file: Path to pid_splits.json
        base_dir: Base directory containing train/val/test image folders
        split_name: Which split to process ('train', 'val', 'test')
        output_file: Output JSON file path
        max_problems: Maximum number of problems to process (None for all)
    """
    
    print(f"Loading ScienceQA data for {split_name} split...")
    
    # Load problems and splits
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    split_pids = splits[split_name]
    print(f"Found {len(split_pids):,} problems in {split_name} split")
    
    # Separate problems with and without images
    problems_with_images = []
    problems_without_images = []
    
    for pid in split_pids:
        if pid not in problems:
            continue
        
        problem = problems[pid]
        has_image = problem.get('image') and problem['image'] and problem['image'] != 'null'
        
        if has_image:
            problems_with_images.append((pid, problem))
        else:
            problems_without_images.append((pid, problem))
    
    print(f"Problems with images: {len(problems_with_images):,}")
    print(f"Problems without images: {len(problems_without_images):,}")
    
    # Process problems with images (group by image)
    print("\nProcessing problems with images...")
    image_to_problems = defaultdict(list)
    
    for pid, problem in problems_with_images:
        # Image path: images/{split}/{pid}/{image_name}
        image_name = problem['image']
        image_path = os.path.join(base_dir, 'images', split_name, pid, image_name)
        
        # Use absolute path
        absolute_image_path = os.path.abspath(image_path)
        
        image_to_problems[absolute_image_path].append((pid, problem))
    
    print(f"Found {len(image_to_problems):,} unique images")
    
    # Process image-based problems
    prompt = "Answer with the option's letter from the given choices directly."
    
    # Process images (grouped by image path)
    image_paths = sorted(image_to_problems.keys())
    if max_problems:
        # Limit by number of problems, not images
        total_with_images = len(problems_with_images)
        if max_problems < total_with_images:
            # Process proportionally
            ratio = max_problems / total_with_images
            image_paths = image_paths[:int(len(image_paths) * ratio)]
    
    # Save output as JSONL
    print(f"\nSaving image-based problems to {output_file_images}...")
    os.makedirs(os.path.dirname(output_file_images), exist_ok=True)
    
    entry_id = 0
    total_qa_pairs_images = 0
    with open(output_file_images, 'w', encoding='utf-8') as f:
    for image_path in tqdm(image_paths, desc="Processing images"):
        problem_list = image_to_problems[image_path]
        
        # Sort by PID for consistency
        problem_list.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        
            # Get relative path starting from "data/" directory
            if "/data/" in image_path:
                data_index = image_path.find("/data/") + 1  # +1 to include the leading /
                relative_path = image_path[data_index:]
            else:
                # Fallback: construct path assuming standard structure
                relative_path = f"data/scienceqa/{os.path.basename(os.path.dirname(os.path.dirname(image_path)))}/{os.path.basename(os.path.dirname(image_path))}/{os.path.basename(image_path)}"
            
            # Get image dimensions (use absolute path for file access)
        width, height = get_image_dimensions(image_path)
        
        # Build conversations
        conversations = []
        
        for idx, (pid, problem) in enumerate(problem_list):
            question = problem['question']
                hint = problem.get('hint', '') if problem.get('hint') else None
            choices = problem['choices']
            answer_idx = problem['answer']
            
                # Format answer choices as letters (A, B, C, D, E)
            formatted_choices = format_answer_choices(choices)
                choice_txt = '\n'.join(formatted_choices)
            
                # Get answer as letter (A, B, C, D, E)
                answer = get_answer_letter(answer_idx)
                
                # Build question following evaluation format: hint + '\n' + question + '\n' + choices + '\n' + prompt
                if hint is not None:
                    question_text = hint + '\n' + question
                else:
                    question_text = question
                
                question_text += '\n' + choice_txt
            
                # First question includes image tag and prompt (with '\n' separator to match evaluation)
            if idx == 0:
                    human_value = f"<image>\n{question_text}\n{prompt}"
            else:
                # Subsequent questions don't include the prompt
                    human_value = question_text
            
            conversations.append({
                "from": "human",
                "value": human_value
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })
        
        # Create entry with image
        entry = {
                "id": entry_id,
                "image": relative_path,
            "width": width,
            "height": height,
            "conversations": conversations
        }
        
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            entry_id += 1
            total_qa_pairs_images += len(conversations) // 2
    
    # Process problems without images (standard InternVL format, no image field)
    print("\nProcessing problems without images...")
    
    if max_problems:
        # Adjust remaining count
        remaining = max_problems - entry_id
        if remaining > 0:
            problems_without_images = problems_without_images[:remaining]
        else:
            problems_without_images = []
    
    # Save output as JSONL
    print(f"\nSaving text-only problems to {output_file_text}...")
    os.makedirs(os.path.dirname(output_file_text), exist_ok=True)
    
    text_entry_id = 0
    total_qa_pairs_images = 0
    total_qa_pairs_text = 0
    
    with open(output_file_text, 'w', encoding='utf-8') as f:
    for pid, problem in tqdm(problems_without_images, desc="Processing text-only"):
        question = problem['question']
            hint = problem.get('hint', '') if problem.get('hint') else None
        choices = problem['choices']
        answer_idx = problem['answer']
        
            # Format answer choices as letters (A, B, C, D, E)
        formatted_choices = format_answer_choices(choices)
            choice_txt = '\n'.join(formatted_choices)
        
            # Get answer as letter (A, B, C, D, E)
            answer = get_answer_letter(answer_idx)
            
            # Build question following evaluation format: hint + '\n' + question + '\n' + choices + '\n' + prompt
            if hint is not None:
                question_text = hint + '\n' + question
            else:
                question_text = question
            
            question_text += '\n' + choice_txt
            question_text += '\n' + prompt
        
        # Standard InternVL format (no image field)
            human_value = question_text
        
        conversations = [
            {
                "from": "human",
                "value": human_value
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
        
        entry = {
                "id": text_entry_id,
            "conversations": conversations
        }
        
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            text_entry_id += 1
            total_qa_pairs_text += 1
    
    print(f"\nDone!")
    print(f"Image-based entries: {entry_id}")
    print(f"  Total QA pairs: {total_qa_pairs_images:,}")
    print(f"Text-only entries: {text_entry_id}")
    print(f"  Total QA pairs: {total_qa_pairs_text:,}")
    print(f"Total entries: {entry_id + text_entry_id}")
    print(f"Total QA pairs: {total_qa_pairs_images + total_qa_pairs_text:,}")


def main():
    parser = argparse.ArgumentParser(description="Process ScienceQA dataset for InternVL training")
    parser.add_argument(
        "--problems_file",
        type=str,
        default="/media/data2/maschan/internvl/data/scienceqa/problems.json",
        help="Path to problems.json file"
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="/media/data2/maschan/internvl/data/scienceqa/pid_splits.json",
        help="Path to pid_splits.json file"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/scienceqa",
        help="Base directory containing train/val/test folders"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to process"
    )
    parser.add_argument(
        "--output_file_images",
        type=str,
        default="/media/data2/maschan/internvl/data/scienceqa/train_images.jsonl",
        help="Output JSONL file path for image-based problems"
    )
    parser.add_argument(
        "--output_file_text",
        type=str,
        default="/media/data2/maschan/internvl/data/scienceqa/train_text.jsonl",
        help="Output JSONL file path for text-only problems"
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_scienceqa(
        problems_file=args.problems_file,
        splits_file=args.splits_file,
        base_dir=args.base_dir,
        split_name=args.split_name,
        output_file_images=args.output_file_images,
        output_file_text=args.output_file_text,
        max_problems=args.max_problems
    )


if __name__ == "__main__":
    main()

