"""
GeoNLI: A flexible, pip-installable toolkit for remote-sensing
Captioning, Grounding, and VQA with pluggable models & datasets.
"""

__version__ = "0.1.0"

from geonli.core.base import (
    VLMBase,
    TransformersVLMBase,
    SegmenterBase,
    TaskBase,
    AgentBase,
    GeoNLIPipeline,
    TaskResult,
    SegmentationResult,
    Detection,
    GeoNLIDataset,
)
from geonli.core.registry import (
    register_vlm,
    register_segmenter,
    register_task,
    register_dataset,
    register_prompt,
    get_vlm,
    get_segmenter,
    get_task,
    get_dataset,
    get_prompt,
    list_registry,
)
from geonli.core.config import ExperimentConfig

# Trigger registration side-effects so users see built-in models/tasks/datasets
import geonli.models    # noqa: F401
import geonli.tasks     # noqa: F401
import geonli.datasets  # noqa: F401

# Optional: trigger adapter imports only if ISRO code is available
try:
    import geonli.adapters  # noqa: F401
except Exception:
    pass

__all__ = [
    "VLMBase",
    "TransformersVLMBase",
    "SegmenterBase",
    "TaskBase",
    "AgentBase",
    "GeoNLIPipeline",
    "TaskResult",
    "SegmentationResult",
    "Detection",
    "GeoNLIDataset",
    "register_vlm",
    "register_segmenter",
    "register_task",
    "register_dataset",
    "register_prompt",
    "get_vlm",
    "get_segmenter",
    "get_task",
    "get_dataset",
    "get_prompt",
    "list_registry",
    "ExperimentConfig",
]
