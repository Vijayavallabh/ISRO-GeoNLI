#!/usr/bin/env python3
"""
VRSBench Benchmark for SAM3 Agent with Direct VLM Loading
Simplified version without run_single_image_inference
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from functools import partial
from tqdm import tqdm
from typing import Dict, List
import argparse
from PIL import Image

# SAM3 imports
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Custom VLM loader
from custom_vlm_loader import DirectVLMLoader, send_generate_request_custom


def parse_vrsbench_bbox(bbox_list: List) -> np.ndarray:
    """Convert polygon/corner list to [x1, y1, x2, y2] format"""
    if len(bbox_list) == 4:
        return np.array(bbox_list)
    
    coords = []
    for i in range(0, len(bbox_list), 2):
        if i+1 < len(bbox_list):
            coords.append([bbox_list[i], bbox_list[i+1]])
    
    if len(coords) >= 2:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return np.array([min(xs), min(ys), max(xs), max(ys)])
    return None


def denormalize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    return np.array([
        bbox[0] * img_width,
        bbox[1] * img_height,
        bbox[2] * img_width,
        bbox[3] * img_height
    ])


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def evaluate_prediction(pred_boxes, gt_bbox: np.ndarray, 
                       img_width: int, img_height: int) -> Dict:
    if pred_boxes is None or len(pred_boxes) == 0:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'detected': False}
    
    gt_bbox_denorm = denormalize_bbox(gt_bbox, img_width, img_height)
    
    best_iou = 0.0
    for pred_box in pred_boxes:
        if torch.is_tensor(pred_box):
            pred_box = pred_box.cpu().numpy()
        
        iou = compute_iou(pred_box, gt_bbox_denorm)
        best_iou = max(best_iou, iou)
    
    # Convert numpy types to native Python types for JSON serialization
    return {
        'iou': float(best_iou),
        'precision': float(1.0 if best_iou > 0.5 else 0.0),
        'recall': float(1.0 if best_iou > 0.5 else 0.0),
        'detected': bool(best_iou > 0.5)
    }


def run_benchmark(
    data_root: str,
    output_dir: str,
    llm_model: str,
    bpe_path: str,
    confidence_threshold: float = 0.5,
    max_samples: int = None
):
    annotations_dir = Path(data_root) / "Annotations_val"
    images_dir = Path(data_root) / "Images_val"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Initializing SAM3...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("✓ SAM3 initialized")
    
    print("\n" + "="*60)
    print("Loading VLM model directly...")
    vlm_loader = DirectVLMLoader(model_name=llm_model, device="cuda")
    print("✓ VLM loaded")
    
    annotation_files = sorted(list(annotations_dir.glob("*.json")))
    if max_samples:
        annotation_files = annotation_files[:max_samples]
    
    print(f"\n✓ Found {len(annotation_files)} annotations to process")
    
    results = []
    metrics_sum = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'detected': 0}
    
    print("\n" + "="*60)
    print("Running benchmark...")
    
    for ann_file in tqdm(annotation_files, desc="Processing samples"):
        try:
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            image_name = annotation.get('image')
            if not image_name:
                print(f"\nWarning: 'image' key missing in {ann_file.name}")
                continue
                
            image_path = images_dir / image_name
            
            if not image_path.exists():
                print(f"\nWarning: Image not found: {image_path}")
                continue
            
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Handle objects list structure
            objects_list = annotation.get('objects', [])
            if not objects_list:
                print(f"\nWarning: No 'objects' list found in {ann_file.name}")
                continue
            
            # Process each object in the annotation
            for obj_info in objects_list:
                referring_sentence = obj_info.get('referring_sentence')
                if not referring_sentence:
                    continue
                
                # Get bbox from obj_corner or obj_coord
                gt_bbox_raw = obj_info.get('obj_corner')
                if not gt_bbox_raw:
                    gt_bbox_raw = obj_info.get('obj_coord')
                
                if not gt_bbox_raw:
                    continue
                
                gt_bbox = parse_vrsbench_bbox(gt_bbox_raw)
                if gt_bbox is None:
                    continue
                
                obj_id = obj_info.get('obj_id', 'unknown')
                
                # Direct SAM3 inference without run_single_image_inference
                inference_state = processor.set_image(img)
                pred_output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=referring_sentence
                )
                pred_boxes = pred_output.get("boxes")
                
                # Evaluate
                metrics = evaluate_prediction(pred_boxes, gt_bbox, img_width, img_height)
                
                result = {
                    'image': image_name,
                    'annotation_file': ann_file.name,
                    'obj_id': int(obj_id) if isinstance(obj_id, (np.integer, np.int64)) else obj_id,
                    'referring_sentence': referring_sentence,
                    'gt_bbox': [float(x) for x in gt_bbox.tolist()],
                    'metrics': metrics
                }
                results.append(result)
                
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]
            
        except Exception as e:
            print(f"\nError processing {ann_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    n_samples = len(results)
    if n_samples > 0:
        avg_metrics = {key: value / n_samples for key, value in metrics_sum.items()}
    else:
        avg_metrics = metrics_sum
    
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'avg_metrics': avg_metrics,
            'total_samples': n_samples,
            'per_sample_results': results
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print(f"Total objects evaluated: {n_samples}")
    print(f"Average IoU: {avg_metrics['iou']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"Detection Rate: {avg_metrics['detected']:.4f}")
    print("="*60)
    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./benchmark_output')
    parser.add_argument('--llm_model', type=str, default='Qwen/Qwen3-VL-8B-Thinking')
    parser.add_argument('--bpe_path', type=str, default='sam3_repo/assets/bpe_simple_vocab_16e6.txt.gz')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=500)
    
    args = parser.parse_args()
    
    run_benchmark(
        args.data_root, args.output_dir, args.llm_model, 
        args.bpe_path, args.confidence_threshold, args.max_samples
    )

if __name__ == "__main__":
    main()