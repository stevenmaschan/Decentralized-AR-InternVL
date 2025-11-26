#!/usr/bin/env python3
"""
Create dataset mixture JSON file describing all datasets.

This script creates a JSON file with dataset configuration including:
- root: path to images
- annotation: path to JSONL annotation file
- data_augment: false
- max_dynamic_patch: maximum number of patches
- repeat_time: repeat factor from statistics
- length: number of samples in the dataset
"""

import json
import os
import argparse


def count_jsonl_lines(file_path):
    """Count number of lines in a JSONL file."""
    if not os.path.exists(file_path):
        return 0
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Create dataset mixture JSON file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dataset_mixture.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/data2/maschan/internvl/data",
        help="Root directory for data"
    )
    
    args = parser.parse_args()
    
    # Dataset configuration
    # Based on dataset statistics, repeat times are:
    # ai2d: 4, aokvqa: 2, chartqa: 4, docvqa: 4, infographicsvqa: 4, scienceqa_image: 4, textvqa: 4, others: 1
    
    datasets_config = {
        "ai2d": {
            "root": os.path.join(args.data_root, "ai2diagram/images"),
            "annotation": os.path.join(args.data_root, "ai2diagram/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0  # Will be computed
        },
        "aokvqa": {
            "root": os.path.join(args.data_root, "coco/train2017"),
            "annotation": os.path.join(args.data_root, "aokvqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 2,
            "length": 0
        },
        "chartqa": {
            "root": os.path.join(args.data_root, "chartqa/ChartQA Dataset/train/png"),
            "annotation": os.path.join(args.data_root, "chartqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0
        },
        "docvqa": {
            "root": os.path.join(args.data_root, "docvqa/images"),
            "annotation": os.path.join(args.data_root, "docvqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0
        },
        "gqa": {
            "root": os.path.join(args.data_root, "gqa/images"),
            "annotation": os.path.join(args.data_root, "gqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "infographicsvqa": {
            "root": os.path.join(args.data_root, "infographicsvqa/infographicsvqa_images"),
            "annotation": os.path.join(args.data_root, "infographicsvqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0
        },
        "kvqa": {
            "root": os.path.join(args.data_root, "kvqa/raw/KVQAimgs"),
            "annotation": os.path.join(args.data_root, "kvqa/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "refcoco": {
            "root": os.path.join(args.data_root, "coco/train2014"),
            "annotation": os.path.join(args.data_root, "refcoco/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "refcoco+": {
            "root": os.path.join(args.data_root, "coco/train2014"),
            "annotation": os.path.join(args.data_root, "refcoco+/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "refcocog": {
            "root": os.path.join(args.data_root, "coco/train2014"),
            "annotation": os.path.join(args.data_root, "refcocog/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "scienceqa_image": {
            "root": os.path.join(args.data_root, "scienceqa/images/train"),
            "annotation": os.path.join(args.data_root, "scienceqa/train_images.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0
        },
        "sharegpt4o": {
            "root": os.path.join(args.data_root, "sharegpt4o/images"),
            "annotation": os.path.join(args.data_root, "sharegpt4o/sharegpt4o_train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "textcaps": {
            "root": os.path.join(args.data_root, "textcaps/train_images"),
            "annotation": os.path.join(args.data_root, "textcaps/train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "textvqa": {
            "root": os.path.join(args.data_root, "textvqa/train_images"),
            "annotation": os.path.join(args.data_root, "textvqa/textvqa_train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 4,
            "length": 0
        },
        "vqav2": {
            "root": os.path.join(args.data_root, "vqav2/train2014"),
            "annotation": os.path.join(args.data_root, "vqav2/vqav2_train.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        },
        "allava_instruct_sampled": {
            "root": os.path.join(args.data_root, "allava/images"),
            "annotation": os.path.join(args.data_root, "allava/allava_instruct_train_sampled.jsonl"),
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0
        }
    }
    
    # Count samples for each dataset
    print("Counting samples in each dataset...")
    for dataset_name, config in datasets_config.items():
        annotation_file = config["annotation"]
        if os.path.exists(annotation_file):
            length = count_jsonl_lines(annotation_file)
            config["length"] = length
            print(f"  {dataset_name:25s}: {length:,} samples")
        else:
            print(f"  {dataset_name:25s}: WARNING - annotation file not found: {annotation_file}")
            config["length"] = 0
    
    # Convert absolute paths to relative paths (starting from data/)
    # Or keep absolute paths - let's keep absolute paths for now
    # But convert to use relative paths if the user wants
    
    # Save JSON file
    print(f"\nSaving dataset mixture JSON to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(datasets_config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {len(datasets_config)}")
    total_samples = sum(config["length"] for config in datasets_config.values())
    print(f"Total samples: {total_samples:,}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)
    
    # Print dataset details
    print("\nDataset details:")
    for dataset_name, config in sorted(datasets_config.items()):
        print(f"  {dataset_name:25s}: {config['length']:>8,} samples, repeat_time={config['repeat_time']}")


if __name__ == "__main__":
    main()

