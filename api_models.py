"""
API Request/Response Models
Pydantic models for FastAPI endpoints supporting captioning, grounding, and VQA tasks.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


# ============================================================================
# Request/Response Models
# ============================================================================

class ImageMetadata(BaseModel):
    """Optional metadata about the input image."""
    width: Optional[int] = None
    height: Optional[int] = None
    spatial_resolution_m: Optional[float] = None


class InputImage(BaseModel):
    """
    Input image specification supporting three input methods:
    - image_url: HTTP/HTTPS URL to download image
    - image_base64: Base64-encoded image data (with or without data URL prefix)
    - image_path: Local filesystem path to image file
    """
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None  # Local file system path
    metadata: Optional[ImageMetadata] = None


class CaptionQuery(BaseModel):
    """Request for image caption generation."""
    instruction: str = "Generate a detailed caption."


class GroundingQuery(BaseModel):
    """Request for object detection/grounding in image."""
    instruction: str


class AttributeQuery(BaseModel):
    """
    Request for Visual Question Answering (VQA).
    Supports three types:
    - binary: Yes/No questions
    - numeric: Questions about counts/quantities
    - semantic: Questions about attributes/types/colors
    """
    binary: Optional[Dict[str, str]] = None
    numeric: Optional[Dict[str, str]] = None
    semantic: Optional[Dict[str, str]] = None


class Queries(BaseModel):
    """Collection of query types that can be processed together."""
    caption_query: Optional[CaptionQuery] = None
    grounding_query: Optional[GroundingQuery] = None
    attribute_query: Optional[AttributeQuery] = None


class StructuredRequest(BaseModel):
    """
    Structured input format matching query.json schema.
    Used by /process endpoint for explicit query type specification.
    """
    input_image: InputImage
    queries: Queries


class SimpleRequest(BaseModel):
    """
    Simple text query format - will be automatically classified.
    Used by /query endpoint for natural language queries.
    The query will be classified using LLM into appropriate task type.
    """
    query: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None  # Local file system path
