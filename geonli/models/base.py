"""
Example model implementations + abstract adapters.
Users can add new models here or in external packages via the registry.
"""

from typing import Optional
from PIL import Image

from geonli.core.base import VLMBase, SegmenterBase, SegmentationResult
from geonli.core.registry import register_vlm, register_segmenter


# ---------------------------------------------------------------------------
# Dummy / stub implementations (useful for testing without GPUs)
# ---------------------------------------------------------------------------

@register_vlm("dummy")
class DummyVLM(VLMBase):
    """Returns the prompt back (useful for CI / unit testing)."""

    def __init__(self, echo: str = "dummy", **kwargs):
        self._echo = echo

    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0.0, **kwargs) -> str:
        return f"[{self._echo}] {prompt}"

    def model_name(self) -> str:
        return "dummy"


@register_segmenter("dummy")
class DummySegmenter(SegmenterBase):
    """Returns empty segmentation (useful for testing)."""

    def __init__(self, **kwargs):
        pass

    def segment(self, image, text_prompt, **kwargs) -> Optional[SegmentationResult]:
        return SegmentationResult(masks=[], metadata=[], count=0, image_size=image.size)

    def model_name(self) -> str:
        return "dummy"
