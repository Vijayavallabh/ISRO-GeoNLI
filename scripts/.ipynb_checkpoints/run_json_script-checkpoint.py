"""
Script to run the pipeline from a JSON configuration file.
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


def serialize_detections(detections):
    """Convert detections to JSON-serializable format."""
    serializable = []
    for det in detections:
        serializable.append({
            "id": det["id"],
            "label": det["label"],
            "area_m2": det["area_m2"],
            "center_point": det["center_point"],
            "score": det["score"]
        })
    return serializable


def main(json_path="query.json", output_path="output.json"):
    """
    Run pipeline from JSON configuration.
    
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
    
    # Load image
    image_info = data.get("input_image", {})
    image_id = image_info.get("image_id", "image.png")
    image_url = image_info.get("image_url")
    
    image = load_image(image_id, image_url)
    if image is None:
        return
    
    # Get metadata
    metadata = image_info.get("metadata", {})
    gsd = metadata.get("spatial_resolution_m", 1.0)
    print(f"Using GSD: {gsd} m/pixel")
    
    # Initialize pipeline
    try:
        pipeline = RSPipeline()
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        return
    
    # Prepare results
    results = {
        "image_id": image_id,
        "caption": "",
        "grounding_results": [],
        "vqa_answers": {}
    }
    
    queries = data.get("queries", {})
    
    # A. Captioning
    if "caption_query" in queries:
        instruction = queries["caption_query"]["instruction"]
        results["caption"] = pipeline.generate_caption(image, instruction)
        print(f"\nCaption: {results['caption'][:100]}...")
    
    # B. Grounding
    detections = []
    if "grounding_query" in queries:
        instruction = queries["grounding_query"]["instruction"]
        detections = pipeline.ground_objects(image, instruction, gsd=gsd)
        results["grounding_results"] = serialize_detections(detections)
        print(f"\nFound {len(detections)} objects")
    
    # C. VQA
    if "attribute_query" in queries:
        attr_queries = queries["attribute_query"]
        vqa_results = {}
        
        for q_type, q_data in attr_queries.items():
            question = q_data["instruction"]
            
            answer = pipeline.answer_question(
                image=image,
                query=question,
                detections=detections,
                question_type=q_type,
                gsd=gsd
            )
            
            vqa_results[q_type] = {
                "question": question,
                "answer": answer
            }
            print(f"\nVQA ({q_type}): {answer}")
        
        results["vqa_answers"] = vqa_results
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n--- Pipeline Complete. Results saved to {output_path} ---")


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
