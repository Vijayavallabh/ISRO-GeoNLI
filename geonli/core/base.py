"""
Abstract base classes that define the contracts for every component.
Nothing in 'core' depends on PyTorch, Transformers, or any specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


# ---------------------------------------------------------------------------
# Data containers (backend-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single oriented bounding box detection."""
    object_id: str
    obbox: List[float]          # 8-float polygon [x1,y1,...,x4,y4]
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """Output of a segmenter model."""
    masks: List[Any]            # backend-specific mask representation
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    count: int = 0
    image_size: Tuple[int, int] = (0, 0)


@dataclass
class TaskResult:
    """Unified result wrapper for any task."""
    task_name: str
    query: str
    response: Any               # str, List[Detection], float, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model bases
# ---------------------------------------------------------------------------

class VLMBase(ABC):
    """
    Base class for Vision-Language Models.
    Can be backed by a local transformers model or a remote REST API.
    """

    @abstractmethod
    def query(
        self,
        image: Optional[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Query the VLM with an image (or text-only if image is None).

        Args:
            image: PIL Image or None for text-only queries.
            prompt: User prompt / question.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


class TransformersVLMBase(VLMBase):
    """
    Extended interface for VLMs loaded via HuggingFace transformers.
    Exposes the raw model, processor, and device so that advanced
    techniques (tool-calling agents, chat-template routing) can access them.
    """

    model: Any          # raw transformers model
    processor: Any      # raw transformers processor/tokenizer
    device: Any         # torch device or str

    def chat_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate from a list of chat messages using the model's chat template.
        This is the primitive that agents and routers need.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Decoded assistant response string.
        """
        # Subclasses (e.g., Qwen3VL) must implement this because
        # vision-processing (process_vision_info) differs per model family.
        raise NotImplementedError(
            "TransformersVLMBase subclasses must implement chat_generate()."
        )


class SegmenterBase(ABC):
    """
    Base class for segmentation / grounding models.
    Examples: SAM3, SAM2, Grounding-DINO, Mask2Former, etc.
    """

    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Optional[SegmentationResult]:
        """
        Segment objects in the image matching the text prompt.

        Args:
            image: PIL Image.
            text_prompt: Class description (e.g., "building", "red car").

        Returns:
            SegmentationResult or None if nothing found.
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ---------------------------------------------------------------------------
# Agent base (for SAM-path VQA)
# ---------------------------------------------------------------------------

class AgentBase(ABC):
    """Base for multi-step tool-calling agents used in VQA SAM-path."""

    @abstractmethod
    def run(
        self,
        image: Image.Image,
        user_query: str,
        gsd: float = 1.0,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        """
        Execute the agent loop.

        Returns:
            {"final_answer": str, "steps_taken": int} or {"error": str}.
        """
        ...


# ---------------------------------------------------------------------------
# Task base
# ---------------------------------------------------------------------------

class TaskBase(ABC):
    """
    Base class for all high-level tasks (Captioning, Grounding, VQA).
    A Task receives models through its constructor (Dependency Injection).
    """

    name: str = "base"

    @abstractmethod
    def run(
        self,
        image: Image.Image,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Execute the task on a single image + query.

        Args:
            image: PIL Image.
            query: Task-specific query string.
            context: Optional shared context (e.g., previous detections, metadata).

        Returns:
            TaskResult.
        """
        ...


# ---------------------------------------------------------------------------
# Dataset base (torch-compatible but not strictly required)
# ---------------------------------------------------------------------------

class GeoNLIDataset(ABC):
    """
    Minimal dataset interface.
    __getitem__ should return a dict with at least:
        {
            "image_id": str,
            "image": PIL.Image,
            "queries": {...},      # same schema as current query.json
            "metadata": {...},
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class GeoNLIPipeline(ABC):
    """
    High-level orchestrator that wires together:
      - one or more Tasks
      - optional shared context between tasks
      - output formatting / serialization
    """

    def __init__(
        self,
        tasks: List[TaskBase],
        vlm: Optional[VLMBase] = None,
        segmenter: Optional[SegmenterBase] = None,
    ):
        self.tasks = tasks
        self.vlm = vlm
        self.segmenter = segmenter

    @abstractmethod
    def run(
        self,
        image: Image.Image,
        queries: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TaskResult]:
        """
        Run all enabled tasks for a single sample.

        Args:
            image: PIL Image.
            queries: Dict matching the GeoNLI query schema.
            metadata: Optional image metadata (GSD, width, height, ...).

        Returns:
            Dict mapping task name -> TaskResult.
        """
        ...
