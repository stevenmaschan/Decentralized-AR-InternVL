#!/usr/bin/env python3
"""
Unified script to combine cluster results from multiple experts for any benchmark.
Replaces the need for separate combine scripts for each benchmark.
"""

import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys

# Try to import torchvision for RefCOCO, but handle gracefully if not available
try:
    from torchvision.ops.boxes import box_iou
    import torch
    HAS_TORCHVISION = True
except (ImportError, RuntimeError):
    HAS_TORCHVISION = False
    import torch

# Add internvl_chat to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'internvl_chat'))

# Benchmark configurations
BENCHMARK_CONFIGS = {
    # VQA-style benchmarks (use VQA score evaluation)
    'vqav2': {
        'metric': 'vqa_score',
        'metric_name': 'VQA Score',
        'dataset_name': 'VQAv2 Val Balanced',
        'input_format': 'json',
        'evaluator': 'TextVQAAccuracyEvaluator',
        'evaluator_path': 'eval.vqa.textvqa_eval',
    },
    'textvqa': {
        'metric': 'vqa_score',
        'metric_name': 'VQA Score',
        'dataset_name': 'TextVQA Val Balanced',
        'input_format': 'json',
        'evaluator': 'TextVQAAccuracyEvaluator',
        'evaluator_path': 'eval.vqa.textvqa_eval',
    },
    'okvqa': {
        'metric': 'vqa_score',
        'metric_name': 'VQA Score',
        'dataset_name': 'OKVQA Val Balanced',
        'input_format': 'json',
        'evaluator': 'TextVQAAccuracyEvaluator',
        'evaluator_path': 'eval.vqa.textvqa_eval',
    },
    # Accuracy-based benchmarks
    'scienceqa': {
        'metric': 'accuracy',
        'metric_name': 'Accuracy',
        'dataset_name': 'ScienceQA Test Balanced',
        'input_format': 'jsonl',
    },
    'ai2d': {
        'metric': 'exact_match_accuracy',
        'metric_name': 'Accuracy',
        'dataset_name': 'AI2D Test Balanced',
        'input_format': 'json',
    },
    'gqa': {
        'metric': 'gqa_accuracy',
        'metric_name': 'Accuracy',
        'dataset_name': 'GQA Testdev Balanced',
        'input_format': 'json',
        'convert_script': 'internvl_chat/eval/vqa/convert_gqa_for_eval.py',
        'eval_script': './data/gqa/eval.py',
    },
    # ANLS-based benchmarks
    'docvqa': {
        'metric': 'anls',
        'metric_name': 'ANLS',
        'dataset_name': 'DocVQA Val Balanced',
        'input_format': 'json',
        'eval_script': 'internvl_chat/eval/vqa/infographicsvqa_eval.py',
    },
    'infovqa': {
        'metric': 'anls',
        'metric_name': 'ANLS',
        'dataset_name': 'InfoVQA Val Balanced',
        'input_format': 'json',
        'eval_script': 'internvl_chat/eval/vqa/infographicsvqa_eval.py',
    },
    # Precision-based benchmarks
    'refcoco': {
        'metric': 'precision',
        'metric_name': 'Precision @ 1',
        'dataset_name': 'RefCOCO',
        'input_format': 'json',
        'needs_split': True,  # Val, TestA, TestB
    },
    'refcocoplus': {
        'metric': 'precision',
        'metric_name': 'Precision @ 1',
        'dataset_name': 'RefCOCO+',
        'input_format': 'json',
        'needs_split': True,  # Val, TestA, TestB
    },
    'refcocog': {
        'metric': 'precision',
        'metric_name': 'Precision @ 1',
        'dataset_name': 'RefCOCOg',
        'input_format': 'json',
        'needs_split': True,  # Val, Test
    },
    # F1-based benchmarks
    'pope': {
        'metric': 'f1',
        'metric_name': 'F1 Score',
        'dataset_name': 'POPE Test',
        'input_format': 'json',
        'eval_script': 'internvl_chat/eval/pope/eval_pope.py',
    },
    # Relaxed accuracy
    'chartqa': {
        'metric': 'relaxed_accuracy',
        'metric_name': 'Relaxed Accuracy',
        'dataset_name': 'ChartQA Test',
        'input_format': 'json',
        'needs_split': True,  # Human, Augmented
    },
    # Special case: MME (handled separately due to complexity)
    'mme': {
        'metric': 'mme_scores',
        'metric_name': 'Overall Score',
        'dataset_name': 'MME Test Balanced',
        'input_format': 'directory',
    },
}


def load_results_file(file_path: str, input_format: str) -> List[Dict]:
    """Load results from JSON or JSONL file."""
    if input_format == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif input_format == 'jsonl':
        results = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def save_results_file(results: List[Dict], output_path: str, input_format: str):
    """Save results to JSON or JSONL file."""
    if input_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif input_format == 'jsonl':
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def combine_results(cluster_files: List[str], output_file: str, input_format: str) -> List[Dict]:
    """Combine results from multiple cluster files."""
    all_results = []
    
    for i, cluster_file in enumerate(cluster_files):
        print(f"Loading cluster {i} results from {cluster_file}...")
        cluster_results = load_results_file(cluster_file, input_format)
        all_results.extend(cluster_results)
        print(f"  Loaded {len(cluster_results)} results")
    
    print(f"\nCombined total: {len(all_results)} results")
    
    # Save combined results
    print(f"Saving combined results to {output_file}...")
    save_results_file(all_results, output_file, input_format)
    
    return all_results


def evaluate_vqa_score(combined_results: List[Dict], annotation_file: str) -> float:
    """Evaluate VQA score using TextVQA evaluator."""
    from eval.vqa.textvqa_eval import TextVQAAccuracyEvaluator
    
    evaluator = TextVQAAccuracyEvaluator()
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)['annotations']
    
    # Create question_id to answers mapping
    question_id2answers = {}
    for item in annotations:
        question_id = item['question_id']
        answers = [answer['answer'] for answer in item['answers']]
        question_id2answers[question_id] = answers
    
    # Prepare evaluation data
    eval_data = []
    for item in combined_results:
        question_id = item.get('question_id') or item.get('id')
        if question_id in question_id2answers:
            eval_data.append({
                'pred_answer': item['answer'],
                'gt_answers': question_id2answers[question_id]
            })
    
    # Calculate accuracy
    accuracy = evaluator.eval_pred_list(eval_data)
    return accuracy


def evaluate_accuracy(combined_results: List[Dict], exact_match: bool = True) -> float:
    """Calculate accuracy from combined results (for ScienceQA)."""
    correct = 0
    total = len(combined_results)
    
    for item in combined_results:
        # ScienceQA: direct comparison without lowercasing
        pred = item.get('answer', '')
        gt = item.get('gt_answers', '')
        
        if pred == gt:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_exact_match_accuracy(entries: List[Dict]) -> float:
    """Calculate exact match accuracy (for AI2D)."""
    scores = []
    for elem in entries:
        annotation = elem.get('annotation')
        answer = elem.get('answer', '').strip().lower()
        
        if isinstance(annotation, str):
            score = 1.0 if answer == annotation.strip().lower() else 0.0
        else:
            score = 1.0 if answer == str(annotation).strip().lower() else 0.0
        scores.append(score)
    
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0


def evaluate_gqa_accuracy(combined_file: str, convert_script: str = 'internvl_chat/eval/vqa/convert_gqa_for_eval.py') -> Optional[float]:
    """Evaluate GQA accuracy using external evaluation script."""
    # Convert results to GQA format
    dst_file = './data/gqa/testdev_balanced_predictions.json'
    convert_cmd = f'python {convert_script} --src {combined_file} --dst {dst_file}'
    print(f"Converting results: {convert_cmd}")
    result = subprocess.run(convert_cmd, shell=True, capture_output=True, text=True, 
                           cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"Error converting results: {result.stderr}")
        return None
    
    # Run GQA evaluation
    eval_cmd = 'cd ./data/gqa/ && python eval.py --tier testdev_balanced && cd ../../'
    print(f"Running evaluation: {eval_cmd}")
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True, 
                           cwd=Path(__file__).parent.parent / 'internvl_chat')
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return None
    
    # Extract accuracy from output
    output = result.stdout
    print("Evaluation output:")
    print(output)
    
    # Look for accuracy
    accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', output)
    if accuracy_match:
        accuracy = float(accuracy_match.group(1))
        if accuracy > 1.0:
            accuracy = accuracy / 100.0
        return accuracy
    
    # Try alternative format
    accuracy_match = re.search(r'overall.*?([\d.]+)', output, re.IGNORECASE)
    if accuracy_match:
        accuracy = float(accuracy_match.group(1))
        return accuracy
    
    return None


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness for numeric answers (ChartQA)."""
    try:
        if abs(float(target)) < 1e-6:
            return abs(float(prediction)) < 1e-6
        return abs(float(target) - float(prediction)) / abs(float(target)) <= max_relative_change
    except (ValueError, TypeError):
        return target.strip().lower() == prediction.strip().lower()


def evaluate_relaxed_accuracy(combined_results: List[Dict]) -> float:
    """Calculate relaxed accuracy for ChartQA (handles numeric answers with 5% tolerance)."""
    scores = []
    for elem in combined_results:
        annotation = elem.get('annotation')
        if isinstance(annotation, str):
            annotation = [annotation]
        elif annotation is None:
            annotation = [elem.get('gt_answers', '')]
        
        answer = elem.get('answer', '')
        score = max([
            (1.0 if relaxed_correctness(ann, answer) else 0.0)
            for ann in annotation
        ])
        scores.append(score)
    
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0


def calculate_iou_manual(box1, box2):
    """Manual IoU calculation without torchvision."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def evaluate_precision_refcoco(combined_results: List[Dict]) -> float:
    """Calculate Precision @ 1 for RefCOCO."""
    
    PATTERN = r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]'
    
    correct = 0
    total = 0
    
    for output in combined_results:
        predict_bbox = re.findall(PATTERN, output.get('answer', ''))
        try:
            predict_bbox = (float(predict_bbox[0][0]), float(predict_bbox[0][1]), 
                           float(predict_bbox[0][2]), float(predict_bbox[0][3]))
        except:
            predict_bbox = (0., 0., 0., 0.)
        
        target_bbox = torch.tensor(output['gt_bbox'], dtype=torch.float32).view(-1, 4)
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
        
        if predict_bbox.sum() >= 4:
            predict_bbox = predict_bbox / 1000
        
        predict_bbox[:, 0::2] *= output['hw'][1]
        predict_bbox[:, 1::2] *= output['hw'][0]
        
        if HAS_TORCHVISION:
            iou_matrix = box_iou(predict_bbox, target_bbox)
            iou = iou_matrix.item() if iou_matrix.numel() == 1 else iou_matrix[0, 0].item()
        else:
            # Manual IoU calculation
            pred_box = predict_bbox[0].tolist()
            target_box = target_bbox[0].tolist()
            iou = calculate_iou_manual(pred_box, target_box)
        
        total += 1
        
        if iou >= 0.5:
            correct += 1
    
    precision = correct / total if total > 0 else 0.0
    return precision


def evaluate_anls(combined_file: str, gt_file: str, eval_script: str = 'internvl_chat/eval/vqa/infographicsvqa_eval.py') -> Optional[float]:
    """Evaluate ANLS score using external script."""
    # Load GT file to filter combined results to match GT
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    # Get question IDs from GT
    gt_question_ids = set()
    if isinstance(gt_data, dict) and 'data' in gt_data:
        gt_question_ids = {item.get('questionId', item.get('question_id')) for item in gt_data['data']}
    elif isinstance(gt_data, list):
        gt_question_ids = {item.get('questionId', item.get('question_id')) for item in gt_data}
    
    # Filter combined results to only include those in GT
    with open(combined_file, 'r') as f:
        combined_results = json.load(f)
    
    filtered_results = []
    for item in combined_results:
        qid = item.get('question_id') or item.get('questionId')
        if qid in gt_question_ids:
            filtered_results.append(item)
    
    if len(filtered_results) != len(combined_results):
        print(f"Filtered results: {len(filtered_results)} (from {len(combined_results)}) to match GT file")
        # Save filtered results to a temp file
        filtered_file = Path(combined_file).parent / f"{Path(combined_file).stem}_filtered.json"
        with open(filtered_file, 'w') as f:
            json.dump(filtered_results, f)
        combined_file = str(filtered_file)
    
    cmd = [
        'python', eval_script,
        '-g', str(gt_file),
        '-s', str(combined_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    print("\n" + "="*60)
    print("Evaluation Output:")
    print("="*60)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    print("="*60)
    
    # Extract ANLS score
    for line in result.stdout.split('\n'):
        if 'anls' in line.lower() or 'score' in line.lower():
            numbers = re.findall(r'[\d.]+', line)
            if numbers:
                try:
                    score = float(numbers[0])
                    if 0 <= score <= 1:
                        return score
                except:
                    pass
    
    return None


def evaluate_f1_pope(combined_file: str, question_file: str, eval_script: str = 'internvl_chat/eval/pope/eval_pope.py') -> Optional[float]:
    """Evaluate POPE F1 score using external script."""
    eval_cmd = f'python {eval_script} --annotation-dir ./data/pope/coco --question-file {question_file} --result-file {combined_file}'
    print(f"Running: {eval_cmd}")
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True, 
                           cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print(result.stdout)
    
    # Extract F1 score
    f1_match = re.search(r'Overall F1:\s*([\d.]+)', result.stdout)
    if f1_match:
        f1_score = float(f1_match.group(1))
        if f1_score > 1.0:
            f1_score = f1_score / 100.0
        return f1_score
    
    return None


def evaluate_mme_scores(expert_dirs: List[str], output_dir: str) -> Dict[str, Any]:
    """Evaluate MME scores (complex, requires special handling)."""
    # Import MME calculation
    sys.path.insert(0, str(Path(__file__).parent.parent / 'internvl_chat' / 'eval' / 'mme'))
    from calculation import calculate_metrics
    
    expert_dirs = [Path(d) for d in expert_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_type_dict = {
        'Perception': ['existence', 'count', 'position', 'color', 'posters', 'celebrity', 'scene', 'landmark', 'artwork', 'OCR'],
        'Cognition': ['commonsense_reasoning', 'numerical_calculation', 'text_translation', 'code_reasoning']
    }
    
    all_tasks = []
    for tasks in eval_type_dict.values():
        all_tasks.extend(tasks)
    
    # Combine .txt files from all experts
    for task in all_tasks:
        task_file = task + '.txt'
        combined_lines = []
        
        for expert_dir in expert_dirs:
            expert_file = expert_dir / task_file
            if expert_file.exists():
                with open(expert_file, 'r') as f:
                    combined_lines.extend(f.readlines())
        
        if combined_lines:
            output_file = output_dir / task_file
            with open(output_file, 'w') as f:
                f.writelines(combined_lines)
    
    # Calculate scores
    cal = calculate_metrics()
    cal.process_result(str(output_dir))
    
    # Extract scores
    scores = {}
    for eval_type, task_name_list in eval_type_dict.items():
        total_score = 0
        task_scores = {}
        
        for task_name in task_name_list:
            task_txt = output_dir / (task_name + '.txt')
            if not task_txt.exists():
                continue
            
            lines = open(task_txt, 'r').readlines()
            chunk_lines = list(cal.divide_chunks(lines))
            
            img_num = len(chunk_lines)
            task_score = 0
            acc_plus_correct_num = 0
            gts = []
            preds = []
            
            for img_items in chunk_lines:
                assert len(img_items) == 2
                img_correct_num = 0
                
                for img_item in img_items:
                    try:
                        img_name, question, gt_ans, pred_ans = img_item.split('\t')
                    except:
                        continue
                    gt_ans = gt_ans.lower()
                    pred_ans = pred_ans.lower()
                    
                    assert gt_ans in ['yes', 'no']
                    pred_ans = cal.parse_pred_ans(pred_ans)
                    assert pred_ans in ['yes', 'no', 'other']
                    
                    gts.append(gt_ans)
                    preds.append(pred_ans)
                    
                    if gt_ans == pred_ans:
                        img_correct_num += 1
                
                if img_correct_num == 2:
                    acc_plus_correct_num += 1
            
            metric_dict = cal.compute_metric(gts, preds)
            acc_plus = acc_plus_correct_num / img_num if img_num > 0 else 0
            metric_dict['acc_plus'] = acc_plus
            
            for k, v in metric_dict.items():
                if k in ['acc', 'acc_plus']:
                    task_score += v * 100
            
            task_scores[task_name] = task_score
            total_score += task_score
        
        scores[eval_type] = {
            'total': total_score,
            'tasks': task_scores
        }
    
    return scores


def update_results_file(results_file: str, dataset_name: str, metric_name: str, score: float, 
                       label: str = '', split_name: str = ''):
    """Update the results summary file."""
    results_file = Path(results_file)
    
    # Read existing content
    lines = []
    if results_file.exists():
        with open(results_file, 'r') as f:
            lines = f.readlines()
    
    # Build entry name
    entry_name = dataset_name
    if split_name:
        entry_name += f" {split_name}"
    if label:
        entry_name += f" {label}"
    
    # Remove existing entry
    pattern = re.compile(re.escape(entry_name.split()[0]))  # Match dataset name
    lines = [l for l in lines if not (pattern.search(l) and entry_name.split()[0] in l)]
    
    # Find table section
    table_start = None
    for i, line in enumerate(lines):
        if 'Dataset' in line and 'Metric' in line and 'Score' in line:
            table_start = i
            break
    
    if table_start is not None:
        new_lines = []
        new_lines.extend(lines[:table_start + 2])
        new_lines.append(f"{entry_name:<30} | {metric_name:<20} | {score:.4f}\n")
        new_lines.extend(lines[table_start + 2:])
        lines = new_lines
    else:
        lines.append(f"{entry_name:<30} | {metric_name:<20} | {score:.4f}\n")
    
    # Write back
    with open(results_file, 'w') as f:
        f.writelines(lines)
    
    print(f"\nUpdated results file: {results_file}")
    print(f"  {entry_name}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified script to combine cluster results from multiple experts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VQAv2 with 2 experts
  python combine_cluster_results_unified.py \\
    --benchmark vqav2 \\
    --cluster-files cluster0.json cluster1.json \\
    --output combined.json \\
    --annotation annotations.json \\
    --results-file results.txt

  # VQAv2 with 4 experts (supports any number)
  python combine_cluster_results_unified.py \\
    --benchmark vqav2 \\
    --cluster-files cluster0.json cluster1.json cluster2.json cluster3.json \\
    --output combined.json \\
    --annotation annotations.json \\
    --results-file results.txt

  # ScienceQA with 2 experts
  python combine_cluster_results_unified.py \\
    --benchmark scienceqa \\
    --cluster-files cluster0.jsonl cluster1.jsonl \\
    --output combined.jsonl \\
    --results-file results.txt

  # RefCOCO with split name
  python combine_cluster_results_unified.py \\
    --benchmark refcoco \\
    --cluster-files cluster0.json cluster1.json \\
    --output combined.json \\
    --split-name Val \\
    --results-file results.txt

  # RefCOCO+ with split name
  python combine_cluster_results_unified.py \\
    --benchmark refcocoplus \\
    --cluster-files cluster0.json cluster1.json \\
    --output combined.json \\
    --split-name Val \\
    --results-file results.txt

  # RefCOCOg with split name (Val or Test)
  python combine_cluster_results_unified.py \\
    --benchmark refcocog \\
    --cluster-files cluster0.json cluster1.json \\
    --output combined.json \\
    --split-name Val \\
    --results-file results.txt

  # DocVQA with ANLS evaluation
  python combine_cluster_results_unified.py \\
    --benchmark docvqa \\
    --cluster-files cluster0.json cluster1.json \\
    --output combined.json \\
    --gt-file gt.json \\
    --results-file results.txt

  # MME with 2 experts
  python combine_cluster_results_unified.py \\
    --benchmark mme \\
    --expert-dirs expert0_dir expert1_dir \\
    --output-dir combined_dir \\
    --results-file results.txt

  # MME with 4 experts (supports any number)
  python combine_cluster_results_unified.py \\
    --benchmark mme \\
    --expert-dirs expert0_dir expert1_dir expert2_dir expert3_dir \\
    --output-dir combined_dir \\
    --results-file results.txt
        """
    )
    
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=list(BENCHMARK_CONFIGS.keys()),
                       help='Benchmark name')
    parser.add_argument('--cluster-files', type=str, nargs='+',
                       help='Paths to cluster result files (JSON/JSONL)')
    parser.add_argument('--expert-dirs', type=str, nargs='+',
                       help='Paths to expert directories (for MME)')
    parser.add_argument('--output', type=str,
                       help='Path to output combined file')
    parser.add_argument('--output-dir', type=str,
                       help='Path to output directory (for MME)')
    parser.add_argument('--annotation', type=str,
                       help='Path to annotation file (for VQA benchmarks)')
    parser.add_argument('--gt-file', type=str,
                       help='Path to ground truth file (for ANLS benchmarks)')
    parser.add_argument('--question-file', type=str,
                       help='Path to question file (for POPE)')
    parser.add_argument('--split-name', type=str,
                       help='Split name (e.g., Val, TestA, TestB for RefCOCO)')
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to results summary text file')
    parser.add_argument('--label', type=str, default='',
                       help='Optional label (e.g., VitL14, VitB16)')
    
    args = parser.parse_args()
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    
    print("=" * 80)
    print(f"Combining {args.benchmark.upper()} Results")
    print("=" * 80)
    print()
    
    # Handle MME separately (uses directories, not files)
    if args.benchmark == 'mme':
        if not args.expert_dirs or not args.output_dir:
            print("Error: MME requires --expert-dirs and --output-dir")
            return
        
        scores = evaluate_mme_scores(args.expert_dirs, args.output_dir)
        
        perception_score = scores['Perception']['total']
        cognition_score = scores['Cognition']['total']
        overall_score = perception_score + cognition_score
        
        label = args.label or 'VitL14'
        update_results_file(args.results_file, f'MME Test Balanced {label}', 'Overall Score', overall_score)
        update_results_file(args.results_file, f'MME Test Balanced {label} Perception', 'Overall Score', perception_score)
        update_results_file(args.results_file, f'MME Test Balanced {label} Cognition', 'Overall Score', cognition_score)
        
        print(f"\nFinal MME Scores:")
        print(f"  Overall: {overall_score:.4f}")
        print(f"  Perception: {perception_score:.4f}")
        print(f"  Cognition: {cognition_score:.4f}")
        return
    
    # For other benchmarks, combine files
    if not args.cluster_files or not args.output:
        print("Error: --cluster-files and --output are required")
        return
    
    # Combine results
    combined_results = combine_results(args.cluster_files, args.output, config['input_format'])
    
    # Evaluate metric
    score = None
    dataset_name = config['dataset_name']
    if args.split_name:
        dataset_name = f"{dataset_name} {args.split_name}"
    
    if config['metric'] == 'vqa_score':
        if not args.annotation:
            print("Error: --annotation is required for VQA benchmarks")
            return
        score = evaluate_vqa_score(combined_results, args.annotation)
        
    elif config['metric'] == 'accuracy':
        score = evaluate_accuracy(combined_results)
        
    elif config['metric'] == 'exact_match_accuracy':
        score = evaluate_exact_match_accuracy(combined_results)
        
    elif config['metric'] == 'gqa_accuracy':
        convert_script = config.get('convert_script', 'internvl_chat/eval/vqa/convert_gqa_for_eval.py')
        score = evaluate_gqa_accuracy(args.output, convert_script)
        
    elif config['metric'] == 'relaxed_accuracy':
        score = evaluate_relaxed_accuracy(combined_results)
        
    elif config['metric'] == 'precision':
        score = evaluate_precision_refcoco(combined_results)
        
    elif config['metric'] == 'anls':
        if not args.gt_file:
            print("Error: --gt-file is required for ANLS benchmarks")
            return
        eval_script = config.get('eval_script', 'internvl_chat/eval/vqa/infographicsvqa_eval.py')
        score = evaluate_anls(args.output, args.gt_file, eval_script)
        
    elif config['metric'] == 'f1':
        if not args.question_file:
            print("Error: --question-file is required for POPE")
            return
        eval_script = config.get('eval_script', 'internvl_chat/eval/pope/eval_pope.py')
        score = evaluate_f1_pope(args.output, args.question_file, eval_script)
    
    if score is not None:
        label = args.label or ''
        update_results_file(args.results_file, dataset_name, config['metric_name'], score, label, args.split_name or '')
        print(f"\nFinal {config['metric_name']}: {score:.4f}")
    else:
        print("\nFailed to calculate score")


if __name__ == '__main__':
    main()
