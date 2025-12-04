"""
Evaluation script that combines inference and evaluation using eval.py
Processes input JSON, generates predictions, and computes GeoNLI scores
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from inference import GeoNLIInference
from eval import GeoNLIEvaluator, print_scores
import torch


class GeoNLIEvaluationPipeline:
    """
    Complete evaluation pipeline: inference + evaluation
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize evaluation pipeline
        
        Args:
            base_model_name: Base VLM model
            adapter_path: Path to fine-tuned LoRA adapter
            device: Device for inference
        """
        # Initialize inference
        self.inference = GeoNLIInference(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            device=device,
        )
        
        # Initialize evaluator
        self.evaluator = GeoNLIEvaluator()
    
    def evaluate_single_sample(
        self,
        input_json_path: str,
        ground_truth_json_path: str,
        output_json_path: Optional[str] = None,
        log_wandb: bool = False,
        wandb_run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single sample
        
        Args:
            input_json_path: Input JSON with queries (no responses)
            ground_truth_json_path: Ground truth JSON with responses
            output_json_path: Where to save predictions
            log_wandb: Whether to log to Weights & Biases
            wandb_run_name: Name for wandb run
            
        Returns:
            Dictionary of evaluation scores
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {input_json_path}")
        print(f"{'='*60}\n")
        
        # Load input
        with open(input_json_path, 'r') as f:
            input_json = json.load(f)
        
        # Generate predictions
        print("Generating predictions...")
        prediction_json = self.inference.process_queries(input_json)
        
        # Save predictions if requested
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(prediction_json, f, indent=4)
            print(f"\nPredictions saved to: {output_json_path}")
        
        # Load ground truth
        with open(ground_truth_json_path, 'r') as f:
            ground_truth_json = json.load(f)
        
        # Parse predictions and ground truths
        predictions, pred_metadata = self.evaluator.parse_response_json(prediction_json)
        ground_truths, gt_metadata = self.evaluator.parse_response_json(ground_truth_json)
        
        # Evaluate
        print("\nEvaluating predictions...")
        scores = self.evaluator.evaluate_all(
            predictions,
            ground_truths,
            metadata=pred_metadata
        )
        
        # Print results
        print_scores(scores, self.evaluator.weights)
        
        # Log to wandb if requested
        if log_wandb:
            self.evaluator.log_to_wandb(
                scores,
                metadata=pred_metadata,
                predictions=predictions,
                ground_truths=ground_truths,
                project_name="GeoNLI-Evaluation",
                run_name=wandb_run_name or Path(input_json_path).stem
            )
        
        return scores
    
    def evaluate_dataset(
        self,
        input_dir: str,
        ground_truth_dir: str,
        output_dir: Optional[str] = None,
        log_wandb: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate entire dataset
        
        Args:
            input_dir: Directory with input JSONs
            ground_truth_dir: Directory with ground truth JSONs
            output_dir: Directory to save predictions
            log_wandb: Whether to log to wandb
            
        Returns:
            Dictionary mapping sample names to scores
        """
        input_path = Path(input_dir)
        gt_path = Path(ground_truth_dir)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None
        
        # Find all input files
        input_files = sorted(input_path.glob("*_input.json"))
        
        all_scores = {}
        aggregate_scores = {
            'captioning': [],
            'grounding': [],
            'binary': [],
            'numeric': [],
            'semantic': [],
            'final_score': []
        }
        
        print(f"\n{'='*60}")
        print(f"Evaluating {len(input_files)} samples")
        print(f"{'='*60}\n")
        
        for input_file in input_files:
            # Determine file names
            sample_name = input_file.stem.replace("_input", "")
            gt_file = gt_path / f"{sample_name}_response.json"
            
            if not gt_file.exists():
                print(f"Warning: Ground truth not found for {sample_name}, skipping...")
                continue
            
            output_file = output_path / f"{sample_name}_prediction.json" if output_path else None
            
            # Evaluate sample
            try:
                scores = self.evaluate_single_sample(
                    str(input_file),
                    str(gt_file),
                    str(output_file) if output_file else None,
                    log_wandb=log_wandb,
                    wandb_run_name=sample_name,
                )
                
                all_scores[sample_name] = scores
                
                # Aggregate
                for key in aggregate_scores.keys():
                    if key in scores:
                        aggregate_scores[key].append(scores[key])
                
            except Exception as e:
                print(f"Error processing {sample_name}: {e}")
                continue
        
        # Compute average scores
        import numpy as np
        avg_scores = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in aggregate_scores.items()
        }
        
        print(f"\n{'='*60}")
        print("Dataset Average Scores")
        print(f"{'='*60}")
        print_scores(avg_scores, self.evaluator.weights)
        
        # Save aggregate results
        if output_path:
            results_file = output_path / "aggregate_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'per_sample_scores': all_scores,
                    'average_scores': avg_scores
                }, f, indent=4)
            print(f"\nAggregate results saved to: {results_file}")
        
        return all_scores


def main():
    parser = argparse.ArgumentParser(description="GeoNLI Evaluation Pipeline")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file or directory"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Ground truth JSON file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predictions"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GeoNLIEvaluationPipeline(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=args.device,
    )
    
    # Check if input is file or directory
    input_path = Path(args.input)
    gt_path = Path(args.ground_truth)
    
    if input_path.is_file():
        # Single file evaluation
        pipeline.evaluate_single_sample(
            args.input,
            args.ground_truth,
            args.output,
            log_wandb=args.wandb,
        )
    else:
        # Dataset evaluation
        pipeline.evaluate_dataset(
            args.input,
            args.ground_truth,
            args.output,
            log_wandb=args.wandb,
        )


if __name__ == "__main__":
    main()
