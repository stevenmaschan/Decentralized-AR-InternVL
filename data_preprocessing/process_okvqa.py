#!/usr/bin/env python3
"""
Preprocess OKVQA dataset into InternVL-compatible format.

This script:
1. Loads questions and annotations JSON files
2. Matches questions with answers by question_id
3. Formats conversations with prompt: "Answer the question using a single word or phrase."
4. Uses COCO train2014 images
"""

import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from collections import defaultdict, Counter


def format_image_path(image_id):
    """Convert image_id to COCO filename format."""
    return f"COCO_train2014_{image_id:012d}.jpg"


def process_okvqa(
    questions_file,
    annotations_file,
    images_dir,
    output_file,
    prompt="Answer the question using a single word or phrase."
):
    """
    Process OKVQA dataset into InternVL format.
    
    Args:
        questions_file: Path to questions JSON file
        annotations_file: Path to annotations JSON file
        images_dir: Directory containing COCO train2014 images
        output_file: Output JSON file path
        prompt: Prompt to add to questions
    """
    print("Loading questions...")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    questions = questions_data['questions']
    print(f"Loaded {len(questions):,} questions")
    
    # Create question_id to question mapping
    question_map = {q['question_id']: q for q in questions}
    
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    annotations = annotations_data['annotations']
    print(f"Loaded {len(annotations):,} annotations")
    
    # Create question_id to annotation mapping
    annotation_map = {ann['question_id']: ann for ann in annotations}
    
    print("Grouping questions by image...")
    # Group questions by image_id
    questions_by_image = defaultdict(list)
    for question in questions:
        image_id = question['image_id']
        questions_by_image[image_id].append(question)
    
    print(f"Found {len(questions_by_image):,} unique images")
    
    print("Processing dataset...")
    output_data = []
    missing_images = 0
    missing_annotations = 0
    
    for image_id, image_questions in tqdm(questions_by_image.items()):
        # Format image path (relative to root directory)
        image_filename = format_image_path(image_id)
        image_path = f"train2014/{image_filename}"
        
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
        
        for question in image_questions:
            question_id = question['question_id']
            question_text = question['question']
            
            # Get annotation
            if question_id not in annotation_map:
                missing_annotations += 1
                continue
            
            annotation = annotation_map[question_id]
            # Get the most common answer from the answers list
            answers = [ans['answer'] for ans in annotation['answers']]
            answer = Counter(answers).most_common(1)[0][0]
            
            # Add human question (with <image> tag only for first conversation)
            if first_conversation:
                human_value = f"<image>\n{question_text}\n{prompt}"
                first_conversation = False
            else:
                human_value = f"{question_text}\n{prompt}"
            
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
    print(f"Missing annotations: {missing_annotations:,}")
    
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess OKVQA dataset")
    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to OKVQA questions JSON file"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="/media/data2/maschan/internvl/datasets/okvqa/mscoco_train2014_annotations.json",
        help="Path to OKVQA annotations JSON file"
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
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Answer the question using a single word or phrase.",
        help="Prompt to add to questions"
    )
    
    args = parser.parse_args()
    
    process_okvqa(
        questions_file=args.questions_file,
        annotations_file=args.annotations_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        prompt=args.prompt
    )


if __name__ == "__main__":
    main()

