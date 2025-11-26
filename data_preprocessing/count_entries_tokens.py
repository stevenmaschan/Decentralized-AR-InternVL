#!/usr/bin/env python3
"""
Count entries and tokens for all processed datasets.
"""

import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

# Initialize InternVL tokenizer
print("Loading InternVL tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL2-8B",
    trust_remote_code=True,
    use_fast=False
)

datasets = {
    "ai2d": "/media/data2/maschan/internvl/data/ai2diagram/train.jsonl",
    "aokvqa": "/media/data2/maschan/internvl/data/aokvqa/train.jsonl",
    "chartqa": "/media/data2/maschan/internvl/data/chartqa/train.jsonl",
    "docvqa": "/media/data2/maschan/internvl/data/docvqa/train.jsonl",
    "gqa": "/media/data2/maschan/internvl/data/gqa/train.jsonl",
    "infographicsvqa": "/media/data2/maschan/internvl/data/infographicsvqa/train.jsonl",
    "kvqa": "/media/data2/maschan/internvl/data/kvqa/train.jsonl",
    "refcoco": "/media/data2/maschan/internvl/data/refcoco/train.jsonl",
    "refcoco+": "/media/data2/maschan/internvl/data/refcoco+/train.jsonl",
    "refcocog": "/media/data2/maschan/internvl/data/refcocog/train.jsonl",
    "scienceqa": "/media/data2/maschan/internvl/data/scienceqa/train.jsonl",
    "sharegpt4o": "/media/data2/maschan/internvl/data/sharegpt4o/sharegpt4o_train.jsonl",
    "textcaps": "/media/data2/maschan/internvl/data/textcaps/train.jsonl",
    "textvqa": "/media/data2/maschan/internvl/data/textvqa/textvqa_train.jsonl",
    "vqav2": "/media/data2/maschan/internvl/data/vqav2/vqav2_train.jsonl",
}

total_entries = 0
total_qa_pairs = 0
total_tokens = 0

results = []

print("\n" + "="*80)
print("Processing datasets...")
print("="*80)

for dataset_name, file_path in datasets.items():
    try:
        print(f"\nProcessing {dataset_name}...")
        
        # All files are now JSONL
        if not os.path.exists(file_path):
            print(f"  SKIP: File not found: {file_path}")
            continue
        
            data = []
        with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
        
        entries = len(data)
        qa_pairs = 0
        for entry in data:
            conversations = entry.get('conversations', [])
            qa_pairs += len(conversations) // 2
        
        # Count tokens
        tokens = 0
        for entry in tqdm(data, desc=f"  Tokenizing {dataset_name}", leave=False):
            conversations = entry.get('conversations', [])
            for conv in conversations:
                text = conv.get('value', '')
                if text:
                # Tokenize the text
                tokenized = tokenizer.encode(text, add_special_tokens=False)
                tokens += len(tokenized)
        
        total_entries += entries
        total_qa_pairs += qa_pairs
        total_tokens += tokens
        
        results.append({
            'dataset': dataset_name,
            'entries': entries,
            'qa_pairs': qa_pairs,
            'tokens': tokens
        })
        
        print(f"  Entries: {entries:,}")
        print(f"  QA pairs: {qa_pairs:,}")
        print(f"  Tokens: {tokens:,}")
        
    except FileNotFoundError:
        print(f"  ERROR: File not found: {file_path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Dataset':<12} {'Entries':>12} {'QA Pairs':>12} {'Tokens':>15}")
print("-" * 80)
for r in results:
    print(f"{r['dataset']:<12} {r['entries']:>12,} {r['qa_pairs']:>12,} {r['tokens']:>15,}")
print("-" * 80)
print(f"{'TOTAL':<12} {total_entries:>12,} {total_qa_pairs:>12,} {total_tokens:>15,}")
print("="*80)

