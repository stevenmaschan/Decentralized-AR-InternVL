#!/usr/bin/env python3
"""
Extract TextCaps entries from LLaVA mix and convert to InternVL JSONL format.

This script:
1. Finds entries with "Provide a one-sentence caption for the provided image." prompt
2. Converts to InternVL format with relative image paths starting from data/
3. Extracts image dimensions
4. Outputs JSONL file compatible with InternVL training
"""

import json
import os
import argparse
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


def process_textcaps_from_llava(
    llava_file,
    textcaps_images_dir,
    output_file,
    max_entries=None
):
    """
    Extract TextCaps entries from LLaVA mix and convert to InternVL format.
    
    Args:
        llava_file: Path to LLaVA mix JSON file
        textcaps_images_dir: Directory containing TextCaps images
        output_file: Output JSON file path
        max_entries: Maximum number of entries to process (None for all)
    """
    
    print("Loading LLaVA mix file...")
    with open(llava_file, 'r') as f:
        llava_data = json.load(f)
    
    target_prompt = "Provide a one-sentence caption for the provided image."
    
    # Find all entries with the caption prompt
    print("Finding TextCaps entries...")
    textcaps_entries = []
    for item in llava_data:
        if 'conversations' in item:
            for conv in item['conversations']:
                if 'value' in conv and target_prompt in conv['value']:
                    textcaps_entries.append(item)
                    break
    
    print(f"Found {len(textcaps_entries):,} entries with caption prompt")
    
    if max_entries:
        textcaps_entries = textcaps_entries[:max_entries]
        print(f"Processing {len(textcaps_entries):,} entries (limited)")
    
    # Save output as JSONL
    print(f"\nProcessing {len(textcaps_entries)} entries...")
    print(f"Saving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entry_id = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
    for idx, item in enumerate(tqdm(textcaps_entries)):
        # Get image path from LLaVA entry
        llava_image_path = item.get('image', '')
        image_filename = os.path.basename(llava_image_path)
        
            # Build absolute path to TextCaps image for file access
        textcaps_image_path = os.path.join(textcaps_images_dir, image_filename)
        absolute_image_path = os.path.abspath(textcaps_image_path)
        
            # Get relative path starting from "data/" directory
            if "/data/" in absolute_image_path:
                data_index = absolute_image_path.find("/data/") + 1  # +1 to include the leading /
                relative_path = absolute_image_path[data_index:]
            else:
                # Fallback: construct path assuming standard structure
                # TextCaps images are typically in textcaps/train_images
                dir_name = os.path.basename(textcaps_images_dir)
                parent_dir = os.path.basename(os.path.dirname(textcaps_images_dir))
                relative_path = f"data/{parent_dir}/{dir_name}/{image_filename}"
            
            # Get image dimensions (use absolute path for reading)
        width, height = get_image_dimensions(textcaps_image_path)
        
        # Extract conversations (should be 2 turns: human + gpt)
        conversations = []
        for conv in item.get('conversations', []):
            conversations.append({
                "from": conv.get('from', '').lower(),
                "value": conv.get('value', '')
            })
        
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
    
    print(f"\nDone! Processed {entry_id:,} entries")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract TextCaps from LLaVA mix for InternVL")
    parser.add_argument(
        "--llava_file",
        type=str,
        default="/media/data2/maschan/playground/data/llava_v1_5_mix665k.json",
        help="Path to LLaVA mix JSON file"
    )
    parser.add_argument(
        "--textcaps_images_dir",
        type=str,
        default="/media/data2/maschan/internvl/data/textcaps/train_images",
        help="Directory containing TextCaps images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/media/data2/maschan/internvl/data/textcaps/train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Maximum number of entries to process (for testing, None for all)"
    )
    
    args = parser.parse_args()
    
    process_textcaps_from_llava(
        llava_file=args.llava_file,
        textcaps_images_dir=args.textcaps_images_dir,
        output_file=args.output_file,
        max_entries=args.max_entries
    )


if __name__ == "__main__":
    main()



