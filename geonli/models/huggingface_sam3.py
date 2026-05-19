"""
HuggingFace SAM3 segmenter wrapper.
Requires bleeding-edge transformers from GitHub:
    pip install git+https://github.com/huggingface/transformers
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from geonli.core.base import SegmenterBase, SegmentationResult
from geonli.core.registry import register_segmenter


@register_segmenter("hf-sam3")
class HuggingFaceSAM3(SegmenterBase):
    """
    Wraps HuggingFace SAM3 (Segment Anything Model 3).

    Args:
        model_id: e.g. ``"facebook/sam3"``
        device: ``"cuda"`` or ``"cpu"``
        spatial_resolution_m: GSD for area calculations
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
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

        from transformers import Sam3Model, Sam3Processor

        print(f"[HuggingFaceSAM3] Loading {self.model_id} ...")

        auth = {}
        if self._hf_token:
            auth["token"] = self._hf_token

        self.processor = Sam3Processor.from_pretrained(self.model_id, **auth)
        self.model = Sam3Model.from_pretrained(self.model_id, **auth).to(self.device).eval()
        self._is_loaded = True
        print(f"[HuggingFaceSAM3] Loaded on {self.device}.")

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Optional[SegmentationResult]:
        """
        Segment objects using text prompt via SAM3.
        """
        self._load()
        import torch

        img_w, img_h = image.size

        inputs = self.processor(
            images=image,
            text=[text_prompt],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,
            mask_threshold=0.3,
            target_sizes=[(img_h, img_w)]
        )[0]

        masks = results.get("masks")
        scores = results.get("scores")

        if masks is None or len(masks) == 0:
            return None

        masks_list = []
        metadata_list = []

        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            masks_list.append(mask_np)

            score_val = float(score.cpu().numpy()) if hasattr(score, "cpu") else float(score)
            metadata_list.append({
                "mask_id": idx,
                "confidence": score_val,
            })

        return SegmentationResult(
            masks=masks_list,
            metadata=metadata_list,
            count=len(masks_list),
            image_size=(img_w, img_h),
        )

    def model_name(self) -> str:
        return f"hf-sam3:{self.model_id}"
