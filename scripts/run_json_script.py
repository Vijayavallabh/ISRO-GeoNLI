"""
Script to run the pipeline from a JSON configuration file.
Can be used as a command-line script or imported and called from Python/notebooks.
"""

import os
import sys
import json
import argparse
import requests
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pipeline import RSPipeline


def load_image(image_id, image_url=None):
    """Load image from local file or URL."""
    if os.path.exists(image_id):
        print(f"Loading local image: {image_id}")
        return Image.open(image_id).convert("RGB")
    elif image_url:
        print(f"Downloading image from {image_url}...")
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(image_id)
            return image
        except Exception as e:
            print(f"Failed to download image: {e}")
            return None
    else:
        print("Error: No valid image found.")
        return None


def format_grounding_response(detections):
    """
    Convert detections to the required grounding response format.
    
    Expected format: [{"object-id": "string", "obbox": [cx, cy, w, h, angle]}]
    OBB format from pipeline: ((cx, cy), (w, h), angle)
    """
    response = []
    for det in detections:
        # Extract OBB: ((cx, cy), (w, h), angle)
        obb = det.get("obb")
        if obb:
            (cx, cy), (w, h), angle = obb
            response.append({
                "object-id": str(det.get("id", "")),
                "obbox": [float(cx), float(cy), float(w), float(h), float(angle)]
            })
    return response


def process_queries(data, pipeline=None, output_path="output.json"):
    """
    Process queries from a dictionary and return results in generic_response.json format.
    This is the main processing function that can be called from notebooks.
    
    Args:
        data: Dictionary with 'input_image' and 'queries' keys (same format as JSON file)
        pipeline: Optional pre-initialized RSPipeline instance (will create new one if None)
        output_path: Path to save results JSON (optional, can be None to skip saving)
        
    Returns:
        Dictionary with results in generic_response.json format
    """
    # Load image
    image_info = data.get("input_image", {})
    image_id = image_info.get("image_id", "image.png")
    image_url = image_info.get("image_url")
    
    image = load_image(image_id, image_url)
    if image is None:
        return None
    
    # Get metadata and ensure it includes actual image dimensions
    metadata = image_info.get("metadata", {}).copy()
    gsd = metadata.get("spatial_resolution_m", 1.0)
    
    # Update metadata with actual image dimensions
    if image:
        metadata["width"] = image.width
        metadata["height"] = image.height
        if "spatial_resolution_m" not in metadata:
            metadata["spatial_resolution_m"] = gsd
    
    print(f"Using GSD: {gsd} m/pixel")
    
    # Initialize pipeline if not provided
    if pipeline is None:
        try:
            pipeline = RSPipeline(vlm_model_id="Qwen/Qwen3-VL-8B-Instruct")
        except Exception as e:
            print(f"Pipeline initialization failed: {e}")
            return None
    
    # Prepare results in the required format (matching generic_response.json)
    results = {
        "input_image": {
            "image_id": image_id,
            "image_url": image_url or "",
            "metadata": metadata
        },
        "queries": {}
    }
    
    queries = data.get("queries", {})
    detections = []
    
    # A. Captioning
    if "caption_query" in queries:
        caption_query = queries["caption_query"]
        instruction = caption_query.get("instruction", "")
        response = pipeline.generate_caption(image, instruction)
        
        results["queries"]["caption_query"] = {
            "instruction": instruction,
            "response": response
        }
        print(f"\nCaption: {response[:100]}...")
    
    # B. Grounding
    if "grounding_query" in queries:
        grounding_query = queries["grounding_query"]
        instruction = grounding_query.get("instruction", "")
        detections = pipeline.ground_objects(image, instruction, gsd=gsd)
        grounding_response = format_grounding_response(detections)
        
        results["queries"]["grounding_query"] = {
            "instruction": instruction,
            "response": grounding_response
        }
        print(f"\nFound {len(detections)} objects")
    
    # C. VQA (attribute_query)
    if "attribute_query" in queries:
        attr_queries = queries["attribute_query"]
        attribute_query_results = {}
        
        for q_type, q_data in attr_queries.items():
            if q_type in ["binary", "numeric", "semantic"]:
                instruction = q_data.get("instruction", "")
                
                answer = pipeline.answer_question(
                    image=image,
                    query=instruction,
                    detections=detections,
                    question_type=q_type,
                    gsd=gsd
                )
                
                attribute_query_results[q_type] = {
                    "instruction": instruction,
                    "response": answer
                }
                print(f"\nVQA ({q_type}): {answer}")
        
        if attribute_query_results:
            results["queries"]["attribute_query"] = attribute_query_results
    
    # Save results if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n--- Pipeline Complete. Results saved to {output_path} ---")
    else:
        print(f"\n--- Pipeline Complete ---")
    
    return results


def main(json_path="query.json", output_path="output.json"):
    """
    Run pipeline from JSON configuration file.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to save results
    """
    print(f"--- Starting Pipeline from JSON ---")
    
    if not os.path.exists(json_path):
        print(f"Error: Input file '{json_path}' not found.")
        return
    
    # Load configuration
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process using the main processing function
    return process_queries(data, pipeline=None, output_path=output_path)


def run_from_dict(data, output_path="output.json", pipeline=None):
    """
    Run pipeline from a dictionary (useful for notebooks or programmatic calls).
    
    Args:
        data: Dictionary with 'input_image' and 'queries' keys (same format as JSON file)
        output_path: Path to save results JSON
        pipeline: Optional pre-initialized RSPipeline instance (will create new one if None)
        
    Returns:
        Dictionary with results in generic_response.json format
    """
    print(f"--- Starting Pipeline from Dictionary ---")
    
    # Load image
    image_info = data.get("input_image", {})
    image_id = image_info.get("image_id", "image.png")
    image_url = image_info.get("image_url")
    
    image = load_image(image_id, image_url)
    if image is None:
        return None
    
    # Get metadata and ensure it includes actual image dimensions
    metadata = image_info.get("metadata", {}).copy()
    gsd = metadata.get("spatial_resolution_m", 1.0)
    
    # Update metadata with actual image dimensions
    if image:
        metadata["width"] = image.width
        metadata["height"] = image.height
        if "spatial_resolution_m" not in metadata:
            metadata["spatial_resolution_m"] = gsd
    
    print(f"Using GSD: {gsd} m/pixel")
    
    # Initialize pipeline if not provided
    if pipeline is None:
        try:
            pipeline = RSPipeline(vlm_model_id="Qwen/Qwen3-VL-8B-Instruct")
        except Exception as e:
            print(f"Pipeline initialization failed: {e}")
            return None
    
    # Prepare results in the required format (matching generic_response.json)
    results = {
        "input_image": {
            "image_id": image_id,
            "image_url": image_url or "",
            "metadata": metadata
        },
        "queries": {}
    }
    
    queries = data.get("queries", {})
    detections = []
    
    # A. Captioning
    if "caption_query" in queries:
        caption_query = queries["caption_query"]
        instruction = caption_query.get("instruction", "")
        response = pipeline.generate_caption(image, instruction)
        
        results["queries"]["caption_query"] = {
            "instruction": instruction,
            "response": response
        }
        print(f"\nCaption: {response[:100]}...")
    
    # B. Grounding
    if "grounding_query" in queries:
        grounding_query = queries["grounding_query"]
        instruction = grounding_query.get("instruction", "")
        detections = pipeline.ground_objects(image, instruction, gsd=gsd)
        grounding_response = format_grounding_response(detections)
        
        results["queries"]["grounding_query"] = {
            "instruction": instruction,
            "response": grounding_response
        }
        print(f"\nFound {len(detections)} objects")
    
    # C. VQA (attribute_query)
    if "attribute_query" in queries:
        attr_queries = queries["attribute_query"]
        attribute_query_results = {}
        
        for q_type, q_data in attr_queries.items():
            if q_type in ["binary", "numeric", "semantic"]:
                instruction = q_data.get("instruction", "")
                
                answer = pipeline.answer_question(
                    image=image,
                    query=instruction,
                    detections=detections,
                    question_type=q_type,
                    gsd=gsd
                )
                
                attribute_query_results[q_type] = {
                    "instruction": instruction,
                    "response": answer
                }
                print(f"\nVQA ({q_type}): {answer}")
        
        if attribute_query_results:
            results["queries"]["attribute_query"] = attribute_query_results
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n--- Pipeline Complete. Results saved to {output_path} ---")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RS Pipeline from JSON configuration"
    )
    parser.add_argument(
        "--input", "-i",
        default="query.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.json",
        help="Path to save results"
    )
    
    args = parser.parse_args()
    main(args.input, args.output)
