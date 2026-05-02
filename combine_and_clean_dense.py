import json
import os
import glob

# List of common cliche prompt instructions to remove
CLICHES = [
    " Answer the question using a single word or phrase.",
    "Answer the question using a single word or phrase.",
    " Answer with the option's letter from the given choices directly.",
    "Answer with the option's letter from the given choices directly.",
    "Please provide the bounding box coordinate of the region this sentence describes: ",
    "Provide a one-sentence caption for the provided image. Reference OCR token: ",
    "Provide a one-sentence caption for the provided image."
]

def remove_cliches(text):
    for cliche in CLICHES:
        text = text.replace(cliche, "")
    
    # Clean up trailing whitespaces per line while keeping \n (like <image>\n)
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    return '\n'.join(lines).strip()

def main():
    input_dir = 'data/dense'
    output_file = 'data/dense_combined_clean.jsonl'
    
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, '*.jsonl')))
    
    print(f"Starting to combine and clean {len(jsonl_files)} files...")
    
    total_processed = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        for filepath in jsonl_files:
            dataset_name = os.path.basename(filepath)
            print(f"Processing {dataset_name}...")
            with open(filepath, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        
                        # (Optional) Track the source dataset in the JSON object
                        data['dataset_source'] = dataset_name
                        
                        # Clean the prompt
                        if 'conversations' in data:
                            for conv in data['conversations']:
                                if conv.get('from') == 'human':
                                    conv['value'] = remove_cliches(conv['value'])
                        
                        fout.write(json.dumps(data) + '\n')
                        total_processed += 1
                    except json.JSONDecodeError:
                        print(f"Warning: JSON decode error in {dataset_name}")
                        continue
                        
    print(f"\nDone! Combined and cleaned {total_processed} items into {output_file}")

if __name__ == "__main__":
    main()
