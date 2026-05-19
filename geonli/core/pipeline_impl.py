"""
Default pipeline implementation that wires registered tasks together
and maintains shared context (e.g., detections for VQA).
"""

from typing import Any, Dict, List, Optional
from PIL import Image

from geonli.core.base import (
    GeoNLIPipeline,
    TaskBase,
    TaskResult,
    VLMBase,
    SegmenterBase,
)


class DefaultGeoNLIPipeline(GeoNLIPipeline):
    """
    Concrete pipeline that:
      1. Runs captioning (if enabled)
      2. Runs grounding (if enabled) and stores detections in context
      3. Runs VQA (if enabled), passing prior detections as context
    """

    def run(
        self,
        image: Image.Image,
        queries: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TaskResult]:
        results: Dict[str, TaskResult] = {}
        context: Dict[str, Any] = {"metadata": metadata or {}}

        # -- Captioning -----------------------------------------------------
        caption_cfg = queries.get("caption_query")
        if caption_cfg:
            task = self._get_task("captioning")
            if task:
                results["captioning"] = task.run(
                    image,
                    caption_cfg.get("instruction", "Describe the image."),
                    context=context,
                )

        # -- Grounding ------------------------------------------------------
        grounding_cfg = queries.get("grounding_query")
        if grounding_cfg:
            task = self._get_task("grounding")
            if task:
                results["grounding"] = task.run(
                    image,
                    grounding_cfg.get("instruction", ""),
                    context=context,
                )
                # Share detections with downstream tasks
                context["detections"] = results["grounding"].response

        # -- VQA (binary / numeric / semantic) ------------------------------
        attr_cfg = queries.get("attribute_query")
        if attr_cfg:
            vqa_task = self._get_task("vqa")
            if vqa_task:
                for qtype in ("binary", "numeric", "semantic"):
                    sub = attr_cfg.get(qtype)
                    if sub and "instruction" in sub:
                        context["question_type"] = qtype
                        key = f"vqa_{qtype}"
                        results[key] = vqa_task.run(
                            image,
                            sub["instruction"],
                            context=context,
                        )

        return results

    def _get_task(self, name: str) -> Optional[TaskBase]:
        for t in self.tasks:
            if getattr(t, "name", None) == name:
                return t
        return None
