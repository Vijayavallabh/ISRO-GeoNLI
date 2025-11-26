"""
Batch inference script for processing multiple samples efficiently
"""

import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from inference import GeoNLIInference
import argparse


def batch_process(
    input_dir: str,
    output_dir: str,
    adapter_path: Optional[str] = None,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
):
    """
    Process all input JSON files in a directory
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save prediction JSON files
        adapter_path: Path to fine-tuned adapter
        base_model: Base model name
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inference = GeoNLIInference(
        base_model_name=base_model,
        adapter_path=adapter_path,
    )
    
    # Find all input JSON files
    input_files = sorted(input_path.glob("*_input.json"))
    
    print(f"\n{'='*60}")
    print(f"Processing {len(input_files)} files")
    print(f"{'='*60}\n")
    
    for input_file in tqdm(input_files, desc="Processing samples"):
        output_file = output_path / input_file.name.replace("_input", "_prediction")
        
        try:
            inference.process_json_file(
                str(input_file),
                str(output_file),
            )
        except Exception as e:
            print(f"\nError processing {input_file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for GeoNLI")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    
    args = parser.parse_args()
    
    batch_process(
        args.input_dir,
        args.output_dir,
        args.adapter_path,
        args.base_model,
    )
