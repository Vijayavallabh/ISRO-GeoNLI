"""
HuggingFace ``datasets`` loader for GeoNLI.
Maps any HF dataset columns to the GeoNLI dict format via configurable mappings.
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image
from io import BytesIO

from geonli.core.base import GeoNLIDataset
from geonli.core.registry import register_dataset


@register_dataset("huggingface")
class HuggingFaceGeoNLIDataset(GeoNLIDataset):
    """
    Wraps a HuggingFace ``datasets.Dataset``.

    Expected config keys:
      - dataset_name_or_path: HF dataset name (e.g. ``nlphuji/flickr30k``) or local path
      - split: ``"train"``, ``"validation"``, ``"test"`` (default ``"train"``)
      - image_column: name of the image column (default ``"image"``)
      - text_column: name of the instruction/caption column (default ``"caption"``)
      - query_type: one of ``caption``, ``grounding``, ``vqa_binary``, ``vqa_numeric``, ``vqa_semantic``

    The dataset item is expected to contain at least an image (PIL or bytes)
    and a text field.  More complex datasets can override ``_build_queries()``.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        image_column: str = "image",
        text_column: str = "caption",
        query_type: str = "caption",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets loader requires `pip install datasets`."
            ) from e

        self.dataset = load_dataset(
            dataset_name_or_path,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.image_column = image_column
        self.text_column = text_column
        self.query_type = query_type
        self._kwargs = kwargs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        image = self._load_image(row[self.image_column])
        text = row[self.text_column]
        queries = self._build_queries(text)
        return {
            "image_id": str(idx),
            "image": image,
            "queries": queries,
            "metadata": {},
        }

    def _load_image(self, raw) -> Image.Image:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")
        # Some HF datasets store images as dicts with "bytes" key
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(BytesIO(raw["bytes"])).convert("RGB")
        raise ValueError(f"Unsupported image type in dataset: {type(raw)}")

    def _build_queries(self, text: str) -> Dict[str, Any]:
        if self.query_type == "caption":
            return {"caption_query": {"instruction": text}}
        if self.query_type == "grounding":
            return {"grounding_query": {"instruction": text}}
        if self.query_type.startswith("vqa_"):
            subtype = self.query_type.replace("vqa_", "")
            return {
                "attribute_query": {
                    subtype: {"instruction": text}
                }
            }
        return {"caption_query": {"instruction": text}}
