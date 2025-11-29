"""
Task implementations for captioning, grounding, and VQA.
"""
from tasks.captioning import CaptioningTask
from tasks.grounding import GroundingTask
from tasks.vqa import VQATask

__all__ = [
    "CaptioningTask",
    "GroundingTask",
    "VQATask"
]

