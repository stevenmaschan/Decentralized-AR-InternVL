#!/usr/bin/env python3
"""
Extract OCR-VQA entries from llava_mix and save in InternVL format.

This script:
1. Loads llava_mix JSON file
2. Extracts entries with ocr_vqa image paths
3. Changes image paths to only include filename (inside images directory)
4. Saves to a new JSON file
"""

import json
import os
import argparse
from tqdm import tqdm


def extract_ocrvqa(llava_mix_file, output_file, images_dir=None):
    """
    Extract OCR-VQA entries from llava_mix.
    
    Args:
        llava_mix_file: Path to llava_mix JSON file
        output_file: Output JSON file path
        images_dir: Optional directory to check if images exist
    """
    print("Loading llava_mix file...")
    with open(llava_mix_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} total entries")
    
    print("Extracting OCR-VQA entries...")
    ocrvqa_entries = []
    missing_images = 0
    
    for entry in tqdm(data):
        image_path = entry.get('image', '')
        
        # Check if this is an OCR-VQA entry
        if 'ocr' in image_path.lower() or 'ocr_vqa' in image_path.lower():
            # Extract filename from path (e.g., "ocr_vqa/images/140031996X.jpg" -> "140031996X.jpg")
            if '/' in image_path:
                filename = image_path.split('/')[-1]
            else:
                filename = image_path
            
            # Optionally check if image exists
            if images_dir:
                full_image_path = os.path.join(images_dir, filename)
                if not os.path.exists(full_image_path):
                    missing_images += 1
                    continue
            
            # Create new entry with updated image path
            new_entry = {
                "id": entry.get("id", len(ocrvqa_entries)),
                "image": filename,  # Only filename, not full path
                "conversations": entry.get("conversations", [])
            }
            
            # Add width/height if present
            if "width" in entry:
                new_entry["width"] = entry["width"]
            if "height" in entry:
                new_entry["height"] = entry["height"]
            
            ocrvqa_entries.append(new_entry)
    
    print(f"\nExtracted {len(ocrvqa_entries):,} OCR-VQA entries")
    if images_dir:
        print(f"Missing images: {missing_images:,}")
    
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ocrvqa_entries, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Extract OCR-VQA entries from llava_mix")
    parser.add_argument(
        "--llava_mix_file",
        type=str,
        default="/media/data2/maschan/playground/data/llava_v1_5_mix665k.json",
        help="Path to llava_mix JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional: Directory containing images to verify existence"
    )
    
    args = parser.parse_args()
    
    extract_ocrvqa(
        llava_mix_file=args.llava_mix_file,
        output_file=args.output_file,
        images_dir=args.images_dir
    )


if __name__ == "__main__":
    main()

