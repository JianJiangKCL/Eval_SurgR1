#!/usr/bin/env python3
"""
Simplified MCQ Accuracy Evaluation Script for Laparo-VLLM

Evaluates multiple-choice question results from model inference output.
Calculates accuracy, precision, recall, F1-score, Jaccard index, and mAP.
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def normalize_text(text):
    """Normalize text by removing prefixes, MCQ options, and whitespace."""
    # Remove common prefixes
    text = re.sub(r"(?i)the\s+(current\s+)?phase\s+is\s*", "", text)
    text = re.sub(r"(?i)the\s+complete\s+surgical\s+actions?\s+are?\s*:\s*", "", text)
    text = re.sub(r"(?i)the\s+answer\s+is\s*", "", text)
    # Remove MCQ option prefix (e.g., "A.", "F.")
    text = re.sub(r"^[A-Z]\.\s*", "", text)
    return text.lower().replace(" ", "")


def extract_answer_from_cot(text):
    """Extract answer from <answer>...</answer> tags if present."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_mcq_answer(text):
    """Parse MCQ format and extract answer content."""
    # Remove common prefixes
    text = re.sub(r"(?i)the\s+complete\s+surgical\s+actions?\s+are?\s*:\s*", "", text)
    text = re.sub(r"(?i)the\s+surgical\s+phase\s+shown\s+in\s+this\s+image\s+is\s*", "", text)
    text = re.sub(r"(?i)the\s+answer\s+is\s*", "", text)
    
    # Find MCQ patterns (Letter. content)
    mcq_matches = re.findall(r'([A-Z])\.\s*([^;,\n]+)', text)
    if mcq_matches:
        parts = [content.strip().rstrip('.,!?;:') for _, content in mcq_matches]
        return ', '.join(parts)
    
    # Fallback: remove letter prefix
    cleaned = re.sub(r'^[A-Z]\.\s*', '', text.strip()).rstrip('.,!?;:')
    return cleaned if cleaned else text.strip()


def check_match(response, labels):
    """Check if response matches labels (exact or partial)."""
    norm_resp = normalize_text(response)
    norm_label = normalize_text(labels)
    
    # Special case: binary yes/no
    if norm_label in {"yes", "no"} and norm_resp in {"yes", "no"}:
        return norm_resp == norm_label
    
    # Exact match or substring match
    return norm_resp == norm_label or norm_label in norm_resp or norm_resp in norm_label


def calculate_metrics(predictions):
    """Calculate per-class precision, recall, F1, and Jaccard."""
    classes = set(p['true'] for p in predictions)
    metrics = {}
    
    for cls in classes:
        tp = sum(1 for p in predictions if p['pred'] == cls and p['true'] == cls)
        fp = sum(1 for p in predictions if p['pred'] == cls and p['true'] != cls)
        fn = sum(1 for p in predictions if p['pred'] != cls and p['true'] == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        support = sum(1 for p in predictions if p['true'] == cls)
        
        metrics[cls] = {
            'precision': precision, 'recall': recall, 
            'f1': f1, 'jaccard': jaccard, 'support': support
        }
    
    return metrics


def calculate_map(predictions):
    """Calculate mean Average Precision."""
    classes = set(p['true'] for p in predictions)
    aps = []
    
    for cls in classes:
        fn_total = sum(1 for p in predictions if p['true'] == cls)
        tp, fp = 0, 0
        precisions, recalls = [], []
        
        for pred in predictions:
            if pred['pred'] == cls:
                tp += 1 if pred['true'] == cls else 0
                fp += 0 if pred['true'] == cls else 1
                if (tp + fp) > 0:
                    precisions.append(tp / (tp + fp))
                    recalls.append(tp / fn_total if fn_total > 0 else 0)
        
        # 11-point interpolation
        ap = 0.0
        for r in [i/10 for i in range(11)]:
            max_p = max((p for p, rec in zip(precisions, recalls) if rec >= r), default=0)
            ap += max_p
        aps.append(ap / 11)
    
    return sum(aps) / len(aps) if aps else 0.0


def evaluate(jsonl_file):
    """Main evaluation function."""
    predictions = []
    norm_to_orig = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Get response and labels from various possible field names
            response = (data.get("response") or data.get("prediction") or 
                       data.get("predicted_phase") or data.get("output"))
            labels = (data.get("labels") or data.get("label") or 
                     data.get("ground_truth") or data.get("true_phase") or data.get("target"))
            
            if not response or not labels:
                continue
            
            # Handle nested structures
            if isinstance(response, dict):
                response = response.get("text", "") or response.get("value", "")
            if isinstance(labels, dict):
                labels = labels.get("text", "") or labels.get("value", "")
            
            response = str(response).strip()
            labels = str(labels).strip()
            
            # Extract from CoT format and parse MCQ
            response = extract_answer_from_cot(response)
            labels = extract_answer_from_cot(labels)
            response = parse_mcq_answer(response)
            labels = parse_mcq_answer(labels)
            
            # Normalize for comparison
            norm_resp = normalize_text(response)
            norm_label = normalize_text(labels)
            
            # Store display mappings
            clean_resp = re.sub(r"^[A-Z]\.\s*", "", response).strip()
            clean_label = re.sub(r"^[A-Z]\.\s*", "", labels).strip()
            norm_to_orig[norm_resp] = clean_resp
            norm_to_orig[norm_label] = clean_label
            
            is_correct = check_match(response, labels)
            
            predictions.append({
                'true': norm_label, 'pred': norm_resp,
                'true_display': clean_label, 'pred_display': clean_resp,
                'correct': is_correct
            })
    
    return predictions, norm_to_orig


def main():
    parser = argparse.ArgumentParser(description='Evaluate MCQ accuracy from inference results')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: same as input file)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show per-class metrics')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {args.input_file} does not exist")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    predictions, norm_to_orig = evaluate(args.input_file)
    
    if not predictions:
        print("Error: No valid predictions found in the input file")
        return 1
    
    # Calculate metrics
    correct = sum(1 for p in predictions if p['correct'])
    total = len(predictions)
    accuracy = correct / total
    
    class_metrics = calculate_metrics(predictions)
    map_score = calculate_map(predictions)
    
    # Calculate macro and weighted averages
    num_classes = len(class_metrics)
    total_support = sum(m['support'] for m in class_metrics.values())
    
    macro = {k: sum(m[k] for m in class_metrics.values()) / num_classes 
             for k in ['precision', 'recall', 'f1', 'jaccard']}
    
    weighted = {k: sum(m[k] * m['support'] for m in class_metrics.values()) / total_support 
                for k in ['precision', 'recall', 'f1', 'jaccard']} if total_support > 0 else {}
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MCQ Evaluation Results: {input_path.name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"mAP:       {map_score:.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {macro['precision']:.4f}  Recall: {macro['recall']:.4f}")
    print(f"  F1-Score:  {macro['f1']:.4f}  Jaccard: {macro['jaccard']:.4f}")
    
    if weighted:
        print(f"\nWeighted Average:")
        print(f"  Precision: {weighted['precision']:.4f}  Recall: {weighted['recall']:.4f}")
        print(f"  F1-Score:  {weighted['f1']:.4f}  Jaccard: {weighted['jaccard']:.4f}")
    
    # Verbose: per-class metrics
    if args.verbose:
        sorted_classes = sorted(class_metrics.keys(), key=lambda x: norm_to_orig.get(x, x))
        max_len = min(max(len(norm_to_orig.get(c, c)) for c in sorted_classes), 35)
        
        print(f"\n{'Class':{max_len}} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Jacc':>8} {'Supp':>6}")
        print("-" * (max_len + 46))
        
        for cls in sorted_classes:
            m = class_metrics[cls]
            name = norm_to_orig.get(cls, cls)[:max_len]
            print(f"{name:{max_len}} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['jaccard']:>8.4f} {m['support']:>6}")
    
    # Save results
    output_path = output_dir / f"{input_path.stem}_acc.json"
    results = {
        'input_file': str(input_path),
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'map': map_score,
        'macro_average': macro,
        'weighted_average': weighted,
        'num_classes': num_classes
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

