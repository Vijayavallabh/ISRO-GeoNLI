"""
Inference script for fine-tuned Qwen3-VL model on GeoNLI tasks
Handles caption, grounding, and attribute queries from structured JSON input
"""

import json
import torch
import re
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


class GeoNLIInference:
    """
    Inference pipeline for GeoNLI tasks using fine-tuned Qwen3-VL
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "bfloat16",
    ):
        """
        Initialize the inference pipeline
        
        Args:
            base_model_name: Base model identifier
            adapter_path: Path to LoRA adapter weights (if None, uses base model)
            device: Device to run inference on
            torch_dtype: Torch dtype for model weights
        """
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        print(f"\n{'='*60}")
        print(f"Loading GeoNLI Inference Pipeline")
        print(f"{'='*60}\n")
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=2048*28*28,
        )
        
        # Load model
        print(f"Loading base model: {base_model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        
        # Load adapter if provided
        if adapter_path:
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=self.torch_dtype,
            )
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {device}\n")
    
    def load_image(self, image_source: Union[str, Path]) -> Image.Image:
        """
        Load image from file path or URL
        
        Args:
            image_source: File path or URL to image
            
        Returns:
            PIL Image
        """
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content))
        else:
            # Load from file
            image = Image.open(image_source)
        
        return image.convert("RGB")
    
    def generate_response(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 20,
    ) -> str:
        """
        Generate response for a single instruction
        
        Args:
            image: PIL Image
            instruction: Text instruction/query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        # Format prompt for Qwen3-VL
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>\n{instruction}\nAnswer:"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode and extract answer
        full_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract text after "Answer:"
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text.strip()
        
        return answer
    
    def parse_bounding_boxes(self, text: str) -> List[Dict[str, Union[str, List[float]]]]:
        """
        Parse bounding box predictions from text response
        
        Expected format examples:
        - "Object 1: [0.5, 0.17, 0.08, 0.08, -37.18]"
        - "Aircraft at [0.5, 0.17, 0.08, 0.08, -37.18]"
        - "[0.5, 0.17, 0.08, 0.08, -37.18], [0.4, 0.1, 0.09, 0.09, 0]"
        
        Returns:
            List of dicts with object-id and obbox
        """
        boxes = []
        
        # Pattern to match lists of 5 numbers (obbox format)
        pattern = r'\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?'
        
        matches = re.findall(pattern, text)
        
        for idx, match in enumerate(matches, 1):
            try:
                obbox = [float(x) for x in match]
                boxes.append({
                    "object-id": str(idx),
                    "obbox": obbox
                })
            except ValueError:
                continue
        
        return boxes
    
    def parse_numeric_response(self, text: str) -> float:
        """
        Extract numeric value from text response
        
        Args:
            text: Generated text
            
        Returns:
            Extracted numeric value
        """
        # Try to find first number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        return 0.0
    
    def process_queries(
        self,
        input_json: Dict,
        image_path: Optional[str] = None,
    ) -> Dict:
        """
        Process all queries from input JSON and generate responses
        
        Args:
            input_json: Input JSON with queries
            image_path: Optional override for image path
            
        Returns:
            Complete output JSON with responses
        """
        # Load image
        if image_path:
            image_source = image_path
        elif "image_url" in input_json["input_image"]:
            image_source = input_json["input_image"]["image_url"]
        elif "image_id" in input_json["input_image"]:
            image_source = input_json["input_image"]["image_id"]
        else:
            raise ValueError("No image source found in input JSON")
        
        print(f"Loading image from: {image_source}")
        image = self.load_image(image_source)
        
        # Initialize output JSON
        output_json = {
            "input_image": input_json["input_image"],
            "queries": {}
        }
        
        queries = input_json["queries"]
        
        # Process caption query
        if "caption_query" in queries:
            print("\nProcessing caption query...")
            instruction = queries["caption_query"]["instruction"]
            response = self.generate_response(
                image,
                instruction,
                max_new_tokens=256,
                temperature=0.5,
            )
            output_json["queries"]["caption_query"] = {
                "instruction": instruction,
                "response": response
            }
            print(f"Caption: {response[:100]}...")
        
        # Process grounding query
        if "grounding_query" in queries:
            print("\nProcessing grounding query...")
            instruction = queries["grounding_query"]["instruction"]
            
            # Add format hint to instruction
            enhanced_instruction = (
                f"{instruction}\n"
                "Return bounding boxes in format [center_x, center_y, width, height, angle] "
                "where coordinates are normalized (0-1) and angle is in degrees."
            )
            
            response = self.generate_response(
                image,
                enhanced_instruction,
                max_new_tokens=512,
                temperature=0.3,
            )
            
            # Parse bounding boxes
            boxes = self.parse_bounding_boxes(response)
            output_json["queries"]["grounding_query"] = {
                "instruction": instruction,
                "response": boxes
            }
            print(f"Found {len(boxes)} objects")
        
        # Process attribute queries
        if "attribute_query" in queries:
            attr_queries = queries["attribute_query"]
            output_json["queries"]["attribute_query"] = {}
            
            # Binary query
            if "binary" in attr_queries:
                print("\nProcessing binary attribute query...")
                instruction = attr_queries["binary"]["instruction"]
                response = self.generate_response(
                    image,
                    instruction,
                    max_new_tokens=32,
                    temperature=0.1,
                )
                # Normalize to Yes/No
                response_lower = response.lower()
                if "yes" in response_lower:
                    normalized_response = "Yes"
                elif "no" in response_lower:
                    normalized_response = "No"
                else:
                    normalized_response = response
                
                output_json["queries"]["attribute_query"]["binary"] = {
                    "instruction": instruction,
                    "response": normalized_response
                }
                print(f"Binary: {normalized_response}")
            
            # Numeric query
            if "numeric" in attr_queries:
                print("\nProcessing numeric attribute query...")
                instruction = attr_queries["numeric"]["instruction"]
                response = self.generate_response(
                    image,
                    instruction,
                    max_new_tokens=64,
                    temperature=0.1,
                )
                numeric_value = self.parse_numeric_response(response)
                output_json["queries"]["attribute_query"]["numeric"] = {
                    "instruction": instruction,
                    "response": numeric_value
                }
                print(f"Numeric: {numeric_value}")
            
            # Semantic query
            if "semantic" in attr_queries:
                print("\nProcessing semantic attribute query...")
                instruction = attr_queries["semantic"]["instruction"]
                response = self.generate_response(
                    image,
                    instruction,
                    max_new_tokens=128,
                    temperature=0.3,
                )
                output_json["queries"]["attribute_query"]["semantic"] = {
                    "instruction": instruction,
                    "response": response
                }
                print(f"Semantic: {response}")
        
        return output_json
    
    def process_json_file(
        self,
        input_json_path: str,
        output_json_path: str,
        image_path: Optional[str] = None,
    ):
        """
        Process queries from JSON file and save results
        
        Args:
            input_json_path: Path to input JSON file
            output_json_path: Path to save output JSON
            image_path: Optional override for image path
        """
        print(f"\n{'='*60}")
        print(f"Processing JSON file: {input_json_path}")
        print(f"{'='*60}\n")
        
        # Load input JSON
        with open(input_json_path, 'r') as f:
            input_json = json.load(f)
        
        # Process queries
        output_json = self.process_queries(input_json, image_path)
        
        # Save output
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'='*60}\n")


def main():
    """
    Example usage
    """
    # Initialize inference pipeline
    inference = GeoNLIInference(
        base_model_name="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="./qwen3_vl_vrsbench_lora/final_model",  # Set to None to use base model
        device="cuda",
        torch_dtype="bfloat16",
    )
    
    # Process single JSON file
    inference.process_json_file(
        input_json_path="sample_dataset_inter_iit_v1_3/sample1_input.json",
        output_json_path="sample_dataset_inter_iit_v1_3/sample1_prediction.json",
        image_path=None,  # Uses image_url from JSON
    )


if __name__ == "__main__":
    main()
