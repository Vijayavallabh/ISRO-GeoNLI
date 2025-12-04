"""
Helper functions for API endpoints.
Contains image processing, classification, and response formatting utilities.
"""

from typing import Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image
from fastapi import HTTPException
import requests
import re
from rs_pipeline import RSPipeline


# ============================================================================
# Pipeline Singleton Management
# ============================================================================

_pipeline: Optional[RSPipeline] = None


def get_pipeline(vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct") -> RSPipeline:
    """Return a singleton RSPipeline instance (lazy-initialized)."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RSPipeline(vlm_model_id=vlm_model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {e}")
    return _pipeline


# ============================================================================
# Query Classification
# ============================================================================

def classify_query(query: str, image=None, vlm_interface=None) -> Dict[str, Any]:
    """
    LLM-based classifier to determine query type and structure using Qwen VLM.
    Returns a structured Queries object.
    
    Args:
        query: User's natural language query
        image: PIL Image (optional, will use dummy if None)
        vlm_interface: VLM interface instance (optional, will get from pipeline if None)
    
    Returns:
        Dictionary with query type classification
    """
    # Get VLM from pipeline if not provided
    if vlm_interface is None:
        pipeline = get_pipeline()
        vlm_interface = pipeline.vlm
    
    # Classification prompt
    classification_prompt = f"""Classify the following query into ONE of these categories:

1. CAPTION: Queries asking for general description, summary, or explanation of an image
   Examples: "Describe this image", "What do you see?", "Generate a caption", "Explain what's in the image"

2. GROUNDING: Queries asking to locate, find, or identify specific objects/regions in an image
   Examples: "Find all buildings", "Locate vehicles", "Where are the trees?", "Identify water bodies"

3. VQA_BINARY: Yes/No questions about the image
   Examples: "Is there water?", "Are there buildings?", "Does it have trees?", "Can you see vehicles?"

4. VQA_NUMERIC: Questions asking for counts, quantities, or numeric measurements
   Examples: "How many buildings?", "Count the vehicles", "What is the area?", "How many objects?"

5. VQA_SEMANTIC: Questions about attributes, types, colors, or qualitative properties
   Examples: "What color is the roof?", "What type of terrain?", "What kind of building?", "Which season?"

Query: "{query}"

Respond with ONLY ONE of these words: "CAPTION", "GROUNDING", "VQA_BINARY", "VQA_NUMERIC", or "VQA_SEMANTIC". No explanation needed."""

    try:
        # Use actual image if provided, otherwise create dummy
        if image is None:
            image = Image.new('RGB', (100, 100), color='white')
        
        # Use VLM to classify with actual image context
        response = vlm_interface.query(
            image=image,
            prompt=classification_prompt,
            max_tokens=10,
            temperature=0
        ).strip().upper()
        
        # Parse the response
        queries = {
            "caption_query": None,
            "grounding_query": None,
            "attribute_query": None
        }
        
        if "CAPTION" in response:
            queries["caption_query"] = {"instruction": query}
        elif "GROUNDING" in response:
            queries["grounding_query"] = {"instruction": query}
        elif "VQA_BINARY" in response or "BINARY" in response:
            queries["attribute_query"] = {"binary": {"instruction": query}}
        elif "VQA_NUMERIC" in response or "NUMERIC" in response:
            queries["attribute_query"] = {"numeric": {"instruction": query}}
        elif "VQA_SEMANTIC" in response or "SEMANTIC" in response:
            queries["attribute_query"] = {"semantic": {"instruction": query}}
        else:
            # Default to caption if unclear
            queries["caption_query"] = {"instruction": query}
            
    except Exception as e:
        print(f"Warning: LLM classification failed ({e}), falling back to keyword matching")
        # Fallback to keyword matching
        query_lower = query.lower()
        queries = {
            "caption_query": None,
            "grounding_query": None,
            "attribute_query": None
        }
        
        caption_keywords = ["caption", "describe", "what do you see", "explain", "summary"]
        grounding_keywords = ["locate", "find", "where", "position", "bounding box", "identify"]
        binary_keywords = ["is there", "are there", "does it have", "contains"]
        numeric_keywords = ["how many", "count", "area", "size", "number"]
        
        if any(kw in query_lower for kw in caption_keywords):
            queries["caption_query"] = {"instruction": query}
        elif any(kw in query_lower for kw in grounding_keywords):
            queries["grounding_query"] = {"instruction": query}
        elif any(kw in query_lower for kw in binary_keywords):
            queries["attribute_query"] = {"binary": {"instruction": query}}
        elif any(kw in query_lower for kw in numeric_keywords):
            queries["attribute_query"] = {"numeric": {"instruction": query}}
        else:
            queries["caption_query"] = {"instruction": query}
    
    return queries


# ============================================================================
# Image Processing Functions
# ============================================================================

def strip_data_prefix(b64: str) -> str:
    """Strip data URL prefix if present: data:<mime>;base64,<data>"""
    if not b64:
        return ""
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64


def decode_image_from_base64(b64: str) -> Optional[Image.Image]:
    """Decode a base64 image string into a PIL Image."""
    if not b64:
        return None
    try:
        data = base64.b64decode(b64)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode image from base64")


def load_image_from_url(url: str) -> Optional[Image.Image]:
    """Download and load image from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")


def load_image_from_path(path: str) -> Optional[Image.Image]:
    """Load image from local file system path."""
    try:
        from pathlib import Path
        image_path = Path(path)
        
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image file not found: {path}")
        
        if not image_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
        
        return Image.open(image_path).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from path: {str(e)}")


def get_image_from_input(input_image) -> Image.Image:
    """
    Extract PIL Image from InputImage object.
    
    Args:
        input_image: InputImage pydantic model with image_base64, image_url, or image_path
    
    Returns:
        PIL Image object
    """
    if input_image.image_base64:
        b64 = strip_data_prefix(input_image.image_base64)
        return decode_image_from_base64(b64)
    elif input_image.image_url:
        return load_image_from_url(input_image.image_url)
    elif input_image.image_path:
        return load_image_from_path(input_image.image_path)
    else:
        raise HTTPException(status_code=400, detail="No image provided (need image_base64, image_url, or image_path)")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


# ============================================================================
# Response Formatting Functions
# ============================================================================

def format_grounding_response(detections):
    """
    Convert pipeline detections to API response format.
    Expects canonical obbox = 8-point polygon.
    """
    response = []

    for det in detections or []:
        obbox = det.get("obbox")
        if isinstance(obbox, (list, tuple)) and len(obbox) == 8:
            response.append({
                "object-id": str(det.get("id", det.get("object-id", ""))),
                "obbox": [float(v) for v in obbox],
            })

    return response



# ============================================================================
# VQA Output Normalization Functions
# ============================================================================

def normalize_binary_vqa_answer(answer: str) -> str:
    """
    Normalize binary VQA answers to standard "yes" or "no" format.
    
    Handles various formats:
    - true/false, True/False, TRUE/FALSE
    - yes/no, Yes/No, YES/NO
    - 1/0
    - "affirmative"/"negative" variations
    
    Args:
        answer: Raw answer string from VLM/SAM
        
    Returns:
        Standardized "yes" or "no" string
    """
    if not answer:
        return "no"
    
    # Convert to lowercase and strip whitespace
    answer_clean = str(answer).strip().lower()
    
    # Extract first word if multi-word response
    first_word = answer_clean.split()[0] if answer_clean.split() else answer_clean
    
    # Positive patterns
    positive_patterns = [
        "yes", "true", "1", "affirmative", "correct", "indeed",
        "absolutely", "certainly", "definitely", "present", "exists"
    ]
    
    # Negative patterns
    negative_patterns = [
        "no", "false", "0", "negative", "incorrect", "nope",
        "absent", "missing", "none", "not"
    ]
    
    # Check positive patterns
    for pattern in positive_patterns:
        if pattern in answer_clean or first_word == pattern:
            return "yes"
    
    # Check negative patterns
    for pattern in negative_patterns:
        if pattern in answer_clean or first_word == pattern:
            return "no"
    
    # Default to "no" if uncertain
    return "no"


def normalize_numeric_vqa_answer(answer: str):
    """
    Normalize numeric VQA answers to float format.
    
    Handles various formats:
    - "5" -> 5.0
    - "5.5" -> 5.5
    - "approximately 5" -> 5.0
    - "5 buildings" -> 5.0
    - "about 3.2 square kilometers" -> 3.2
    - "zero" -> 0.0
    - "one" -> 1.0
    
    Args:
        answer: Raw answer string from VLM/SAM
        
    Returns:
        Float value extracted from answer, or 0.0 if parsing fails
    """
    if not answer:
        return ""
    
    answer_clean = str(answer).strip().lower()
    
    # Word to number mapping for common words
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
        "none": 0, "no": 0
    }
    
    # Check for number words first
    for word, num in word_to_num.items():
        if word in answer_clean.split():
            return float(num)
    
    # Extract all numbers (integers and floats) from the string
    numbers = re.findall(r'-?\d+\.?\d*', answer_clean)
    
    if numbers:
        try:
            return float(numbers[0])
        except (ValueError, IndexError):
            pass
    
    # Special cases
    if any(word in answer_clean for word in ["none", "zero", "no ", "not any"]):
        return 0.0
    
    # If we can't extract a number, return 0.0
    return answer

def normalize_vqa_answer(answer: str, question_type: str):
    """
    Main normalization function that routes to appropriate normalizer.
    
    Args:
        answer: Raw answer from VQA system
        question_type: One of "binary", "numeric", or "semantic"
        
    Returns:
        Normalized answer (str for binary/semantic, float for numeric)
    """
    if question_type == "binary":
        return normalize_binary_vqa_answer(answer)
    elif question_type == "numeric":
        return normalize_numeric_vqa_answer(answer)
