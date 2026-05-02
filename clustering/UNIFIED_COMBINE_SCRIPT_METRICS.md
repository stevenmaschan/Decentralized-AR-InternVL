# Unified Combine Script - Metrics Documentation

This document lists all metrics handled by `combine_cluster_results_unified.py` and how they match the original separate scripts.

## Metrics Implemented

### 1. VQA Score
**Benchmarks**: VQAv2, TextVQA, OKVQA
**Implementation**: Uses `TextVQAAccuracyEvaluator` from `eval.vqa.textvqa_eval`
**Method**: Matches original scripts exactly
- Loads annotations with question_id to answers mapping
- Evaluates using `evaluator.eval_pred_list()`

### 2. Accuracy (Simple)
**Benchmarks**: ScienceQA
**Implementation**: Direct string comparison (`answer == gt_answers`)
**Method**: Matches `combine_scienceqa_balanced_2experts_results.py`
- No lowercasing or stripping
- Direct equality check

### 3. Exact Match Accuracy
**Benchmarks**: AI2D
**Implementation**: Case-insensitive exact match with type handling
**Method**: Matches `combine_ai2d_balanced_2experts_results.py`
- Handles both string and non-string annotations
- Uses `.strip().lower()` for comparison

### 4. GQA Accuracy
**Benchmarks**: GQA
**Implementation**: External evaluation script
**Method**: Matches `combine_gqa_2experts_results.py`
- Converts results using `convert_gqa_for_eval.py`
- Runs GQA official evaluation script
- Extracts accuracy from output

### 5. Relaxed Accuracy
**Benchmarks**: ChartQA
**Implementation**: Numeric tolerance (5%) + exact match for non-numeric
**Method**: Matches `combine_chartqa_balanced_2experts_results.py`
- Uses `relaxed_correctness()` function
- For numeric: allows 5% relative error
- For non-numeric: exact match (case-insensitive)
- Handles annotation as list or single value

### 6. Precision @ 1
**Benchmarks**: RefCOCO
**Implementation**: IoU-based precision calculation
**Method**: Matches `combine_refcoco_2experts_results.py`
- Parses bounding box from answer text
- Calculates IoU with ground truth
- Precision = IoU >= 0.5
- Handles coordinate normalization

### 7. F1 Score
**Benchmarks**: POPE
**Implementation**: External evaluation script
**Method**: Matches `combine_pope_2experts_results.py`
- Runs `eval_pope.py` script
- Extracts "Overall F1" from output
- Handles percentage to decimal conversion

### 8. ANLS (Average Normalized Levenshtein Similarity)
**Benchmarks**: DocVQA, InfoVQA
**Implementation**: External evaluation script
**Method**: Matches `combine_docvqa_balanced_2experts_results.py` and `combine_infovqa_balanced_2experts_results.py`
- Runs `infographicsvqa_eval.py` script
- Extracts ANLS score from output
- Handles various output formats

### 9. MME Scores
**Benchmarks**: MME
**Implementation**: Complex multi-task evaluation
**Method**: Matches `combine_mme_2experts_results.py`
- Combines task-specific .txt files
- Uses MME calculation module
- Calculates Perception, Cognition, and Overall scores
- Handles acc and acc_plus metrics

## Input Format Support

- **JSON**: VQAv2, TextVQA, OKVQA, AI2D, GQA, ChartQA, RefCOCO, POPE, DocVQA, InfoVQA
- **JSONL**: ScienceQA
- **Directory**: MME (combines multiple .txt files)

## All Metrics Accounted For

✅ VQA Score (VQAv2, TextVQA, OKVQA)
✅ Accuracy (ScienceQA - direct comparison)
✅ Exact Match Accuracy (AI2D - case-insensitive)
✅ GQA Accuracy (external script)
✅ Relaxed Accuracy (ChartQA - numeric tolerance)
✅ Precision @ 1 (RefCOCO - IoU-based)
✅ F1 Score (POPE - external script)
✅ ANLS (DocVQA, InfoVQA - external script)
✅ MME Scores (MME - complex multi-task)

## Usage

See the script's `--help` for detailed usage examples. Each benchmark's specific metric is automatically selected based on the `--benchmark` argument.
