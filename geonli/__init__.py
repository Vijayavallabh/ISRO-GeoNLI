"""
GeoNLI: A flexible, pip-installable toolkit for remote-sensing
Captioning, Grounding, and VQA with pluggable models & datasets.
"""

__version__ = "0.1.0"

from geonli.core.base import (
    VLMBase,
    SegmenterBase,
    TaskBase,
    GeoNLIPipeline,
    TaskResult,
    SegmentationResult,
    Detection,
)
from geonli.core.registry import (
    register_vlm,
    register_segmenter,
    register_task,
    register_dataset,
    get_vlm,
    get_segmenter,
    get_task,
    get_dataset,
)
from geonli.core.config import ExperimentConfig

__all__ = [
    "VLMBase",
    "SegmenterBase",
    "TaskBase",
    "GeoNLIPipeline",
    "TaskResult",
    "SegmentationResult",
    "Detection",
    "register_vlm",
    "register_segmenter",
    "register_task",
    "register_dataset",
    "get_vlm",
    "get_segmenter",
    "get_task",
    "get_dataset",
    "ExperimentConfig",
]
