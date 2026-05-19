"""
Generic HuggingFace SAM / SAM2 / SAM3 segmenter wrapper.
Auto-detects the model type and provides a unified .segment() interface.
Works with any HF segmentation model that accepts (image, points/boxes/text).
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from geonli.core.base import SegmenterBase, SegmentationResult
from geonli.core.registry import register_segmenter


@register_segmenter("hf-sam")
class HuggingFaceSAM(SegmenterBase):
    """
    Wraps HuggingFace SAM-family models.

    Args:
        model_id: e.g. ``"facebook/sam-vit-huge"``, ``"facebook/sam2-hiera-large"``
        device: ``"cuda"`` or ``"cpu"``
        spatial_resolution_m: GSD for area calculations
    """

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        spatial_resolution_m: float = 1.0,
        hf_token_env: str = "HF_TOKEN",
        **kwargs,
    ):
        self.model_id = model_id
        self.device = device
        self.gsd = spatial_resolution_m
        self._hf_token = os.getenv(hf_token_env) or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self._is_loaded = False

    def _load(self):
        if self._is_loaded:
            return

        import torch
        from transformers import AutoProcessor, SamModel

        print(f"[HuggingFaceSAM] Loading {self.model_id} ...")

        auth = {}
        if self._hf_token:
            auth["token"] = self._hf_token

        self.processor = AutoProcessor.from_pretrained(self.model_id, **auth)
        self.model = SamModel.from_pretrained(self.model_id, **auth).to(self.device).eval()
        self._is_loaded = True
        print(f"[HuggingFaceSAM] Loaded on {self.device}.")

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Optional[SegmentationResult]:
        """
        Segment using a dummy point prompt at the image center.
        SAM-family models are primarily point/box prompted; text conditioning
        is not native.  We use the center point as a generic fallback.
        """
        self._load()
        import torch

        w, h = image.size
        # Use image center as a generic prompt point
        input_points = [[[w // 2, h // 2]]]

        inputs = self.processor(image, input_points=input_points, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]

        scores = outputs.iou_scores.squeeze()  # shape (num_masks,)

        if masks is None or len(masks) == 0:
            return None

        # Convert to list format; each element must be a 2D binary mask
        masks_list = []
        for m in masks:
            m_np = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            # SAM outputs boolean or probability masks; squeeze to 2D
            if m_np.ndim == 3:
                m_np = m_np.squeeze(0)
            masks_list.append(m_np.astype(np.float32))

        metadata_list = []

        for idx, (mask, score) in enumerate(zip(masks_list, scores)):
            score_val = float(score.cpu().numpy()) if hasattr(score, "cpu") else float(score)
            metadata_list.append({
                "mask_id": idx,
                "confidence": score_val,
                "coordinates": [],  # OBB extraction happens later in GroundingTask
            })

        return SegmentationResult(
            masks=masks_list,
            metadata=metadata_list,
            count=len(masks_list),
            image_size=(w, h),
        )

    def model_name(self) -> str:
        return f"hf-sam:{self.model_id}"
