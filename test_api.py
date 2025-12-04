"""
Test script for the updated API endpoints.
"""

import requests
import base64
import json
from pathlib import Path

# BASE_URL = "https://a2f7b83599770.notebooks.jarvislabs.net/proxy/8080/"
BASE_URL = "http://127.0.0.1:8000"

PAYLOAD = {
            "input_image": {
                "image_id": "sample_satellite_001",
                "image_path": "/home/ISRO-GeoNLI/sample_image.png",
                "metadata": {
                "width": 1024,
                "height": 1024,
                "spatial_resolution_m": 0.5
                }
            },
            "queries": {
                "caption_query": {
                "instruction": "Generate a detailed caption describing all visible features in this image"
                },
                "grounding_query": {
                "instruction": "Locate the sun in the image."
                },
                "attribute_query": {
                "binary": {
                    "instruction": "Is there any water body visible in this image?"
                },
                "numeric": {
                    "instruction": "How many suns are there in the image?"
                },
                "semantic": {
                    "instruction": "What type of terrain is shown in this image?"
                }
                }
            }
        }

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_structured_endpoint():
    """Test the /process endpoint with structured schema."""
    print("\n" + "="*60)
    print("Testing /process endpoint (Structured Schema)")
    print("="*60)
    
    payload = PAYLOAD
    
    response = requests.post(f"{BASE_URL}/process", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_simple_query_endpoint():
    print("\n" + "="*60)
    print("Testing /query endpoint (Simple Query with Auto-Classification)")
    print("="*60)

    payload = {
        "query": "describe the image to me",
        "image_path": "/home/ISRO-GeoNLI/sample_image.png",
    }

    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")

    data = response.json()

    # ---- Extract ONLY the answer ----
    results = data.get("results", {})

    answer = None

    if "attributes" in results:
        # Numeric / semantic / binary
        for _, v in results["attributes"].items():
            answer = v.get("response")
            break

    elif "caption" in results:
        answer = results["caption"].get("response")

    elif "grounding" in results:
        answer = results["grounding"].get("detections")

    print("\n✅ Extracted Answer:")
    print(data)



def test_legacy_caption_endpoint():
    """Test legacy /caption endpoint."""
    print("\n" + "="*60)
    print("Testing /caption endpoint (Legacy)")
    print("="*60)
    
    # You'll need to replace this with an actual image path
    # image_b64 = encode_image("path/to/your/image.jpg")
    
    payload = {
        "text": "Describe this satellite image in detail",
        # "image": f"data:image/png;base64,{image_b64}"
    }
    
    response = requests.post(f"{BASE_URL}/caption", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("API Test Suite - ISRO-GeoNLI")
    print("="*60)
    print("\nMake sure the server is running:")
    print("  python run_dev.bat")
    print("  OR")
    print("  uvicorn app_dev:app --reload --port 8000")
    print("\n" + "="*60)
    
    # Uncomment the tests you want to run
    test_structured_endpoint()
    # test_simple_query_endpoint()
    # test_legacy_caption_endpoint()
    
    print("\n✨ Add your image paths and uncomment the tests to run them!")
