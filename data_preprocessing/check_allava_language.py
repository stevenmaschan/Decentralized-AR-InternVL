#!/usr/bin/env python3
"""
Check language in ALLaVA-Instruct file.
"""

import json
import sys

instruct_file = '/media/data2/maschan/internvl/datasets/allava/ALLaVA-Instruct-LAION-4V.json'

def has_chinese(text):
    if not text:
        return False
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

print('Loading ALLaVA-Instruct file...', file=sys.stderr, flush=True)

try:
    with open(instruct_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'Total entries: {len(data):,}')
    
    chinese_count = 0
    non_chinese_count = 0
    sample_chinese = []
    sample_english = []
    
    # Check first 10000 entries
    check_limit = min(10000, len(data))
    
    for i, entry in enumerate(data[:check_limit]):
        has_chinese_in_entry = False
        for conv in entry.get('conversations', []):
            value = conv.get('value', '')
            if has_chinese(value):
                has_chinese_in_entry = True
                if len(sample_chinese) < 3:
                    sample_chinese.append(value[:150])
                break
        
        if has_chinese_in_entry:
            chinese_count += 1
        else:
            non_chinese_count += 1
            if len(sample_english) < 3:
                for conv in entry.get('conversations', []):
                    if conv.get('value', ''):
                        sample_english.append(conv.get('value', '')[:150])
                        break
    
    print(f'\nChecked {check_limit:,} entries')
    print(f'Entries with Chinese: {chinese_count:,}')
    print(f'Entries without Chinese: {non_chinese_count:,}')
    print(f'Percentage Chinese: {chinese_count/check_limit*100:.2f}%')
    
    if sample_chinese:
        print('\nSample entries with Chinese:')
        for i, sample in enumerate(sample_chinese, 1):
            print(f'  {i}. {sample}...')
    
    if sample_english:
        print('\nSample entries without Chinese:')
        for i, sample in enumerate(sample_english, 1):
            print(f'  {i}. {sample}...')
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

