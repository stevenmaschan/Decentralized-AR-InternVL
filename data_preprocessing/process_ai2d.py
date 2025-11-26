#!/usr/bin/env python3
"""
Process AI2D dataset into InternVL-compatible JSONL format.

This script:
1. Reads AI2D annotations and questions
2. Filters out test IDs (keeps only training IDs)
3. Groups all questions per image into one conversation
4. Formats answer choices with letter prefixes (A., B., C., D.)
5. Uses letter answers (A, B, C, D) to match test format
6. Adds prompt at the beginning of each conversation
7. Extracts image dimensions
8. Outputs JSONL file compatible with InternVL training
"""

import json
import os
import csv
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


def get_answer_letter(correct_answer_idx):
    """Get the answer as a letter (A, B, C, D) based on the index."""
    if 0 <= correct_answer_idx < 26:
        return chr(65 + correct_answer_idx)  # A, B, C, D, ...
    else:
        return ""


def format_answer_choices(answer_texts):
    """Format answer choices with letter prefixes (A., B., C., D.)."""
    formatted = []
        for i, answer in enumerate(answer_texts):
        letter = chr(65 + i)  # A, B, C, D, ...
        formatted.append(f"{letter}. {answer}")
    return formatted


def load_test_ids(test_ids_file):
    """Load test image IDs from CSV file."""
    test_ids = set()
    if test_ids_file and os.path.exists(test_ids_file):
        print(f"Loading test IDs from {test_ids_file}...")
        with open(test_ids_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Handle both single column and multi-column CSV
                # Take the first column as the ID
                if row:
                    test_id = row[0].strip()
                    # Remove file extension if present
                    if test_id.endswith('.png'):
                        test_id = test_id[:-4]
                    test_ids.add(test_id)
        print(f"Loaded {len(test_ids)} test IDs")
    else:
        print(f"Warning: Test IDs file not found: {test_ids_file}")
        print("Processing all images (no filtering)")
    return test_ids


def process_ai2d(
    annotations_dir,
    questions_dir,
    images_dir,
    output_file,
    test_ids_file=None,
    max_images=None
):
    """
    Process AI2D dataset into InternVL format.
    
    Args:
        annotations_dir: Directory containing annotation JSON files
        questions_dir: Directory containing question JSON files
        images_dir: Directory containing PNG images
        output_file: Output JSONL file path
        test_ids_file: Path to CSV file containing test image IDs (optional)
        max_images: Maximum number of images to process (None for all)
    """
    
    # Load test IDs if provided
    test_ids = load_test_ids(test_ids_file)
    
    print("Loading AI2D data...")
    
    # Get all question files
    question_files = sorted([f for f in os.listdir(questions_dir) if f.endswith('.json')])
    
    # Group questions by image
    image_questions = defaultdict(list)
    
    for q_file in tqdm(question_files, desc="Loading questions"):
        q_path = os.path.join(questions_dir, q_file)
        
        try:
            with open(q_path, 'r') as f:
                q_data = json.load(f)
            
            if 'questions' in q_data and q_data['questions']:
                image_name = q_data.get('imageName', q_file.replace('.json', ''))
                # Remove file extension if present for ID matching
                image_id = image_name
                if image_id.endswith('.png'):
                    image_id = image_id[:-4]
                
                # Skip if this is a test image
                if test_ids and image_id in test_ids:
                    continue
                
                for q_text, q_info in q_data['questions'].items():
                    image_questions[image_name].append({
                        'question': q_text,
                        'answerTexts': q_info.get('answerTexts', []),
                        'correctAnswer': q_info.get('correctAnswer', 0),
                        'abcLabel': q_info.get('abcLabel', False),
                        'questionId': q_info.get('questionId', '')
                    })
        except Exception as e:
            print(f"Warning: Error reading {q_file}: {e}")
    
    print(f"Found {len(image_questions)} training images with questions")
    print(f"Total questions: {sum(len(qs) for qs in image_questions.values()):,}")
    
    # Process each image
    image_names = sorted(image_questions.keys())
    if max_images:
        image_names = image_names[:max_images]
    
    print(f"\nProcessing {len(image_names)} training images...")
    
    # Open output file for writing
    entry_id = 0
    with open(output_file, 'w', encoding='utf-8') as f:
    for image_name in tqdm(image_names):
        questions = image_questions[image_name]
        
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
            question = q_data['question']
            answer_texts = q_data['answerTexts']
            correct_answer_idx = q_data['correctAnswer']
            
                # Get answer as a letter (A, B, C, D) based on the index
                answer = get_answer_letter(correct_answer_idx)
            
                # Format answer choices with letter prefixes (A., B., C., D.)
                formatted_choices = format_answer_choices(answer_texts)
                answer_choices_text = "\n".join(formatted_choices)
            
                # Prompt (only for first question, after the answer choices)
                # Match the test format prompt exactly
                prompt = "Answer with the option's letter from the given choices directly."
            
                # First question includes image tag and prompt after answer choices (with whitespace separator)
            if idx == 0:
                    human_value = f"<image>\n{question}\n{answer_choices_text} {prompt}"
            else:
                # Subsequent questions don't include the prompt
                    human_value = f"{question}\n{answer_choices_text}"
            
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
    print(f"\nDone! Processed {entry_id} training images")
    print(f"Total conversations: {entry_id}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process AI2D dataset for InternVL training")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/ai2d/annotations",
        help="Directory containing AI2D annotation JSON files"
    )
    parser.add_argument(
        "--questions_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/ai2d/questions",
        help="Directory containing AI2D question JSON files"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/media/data2/maschan/internvl/datasets/ai2d/images",
        help="Directory containing AI2D PNG images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/ai2d/ai2d_train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--test_ids_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/ai2d/ai2d_test_ids.csv",
        help="Path to CSV file containing test image IDs (one per line)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_ai2d(
        annotations_dir=args.annotations_dir,
        questions_dir=args.questions_dir,
        images_dir=args.images_dir,
        output_file=args.output_file,
        test_ids_file=args.test_ids_file,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

