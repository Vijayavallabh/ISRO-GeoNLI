"""
CSV/TSV dataset loader.
Each row corresponds to one sample.  The image path, instruction, and
optional metadata are read from named columns.
"""

import csv
import os
from typing import Any, Dict, List, Optional
from PIL import Image

from geonli.core.base import GeoNLIDataset
from geonli.core.registry import register_dataset


@register_dataset("csv_dataset")
class CSVGeoNLIDataset(GeoNLIDataset):
    """
    Loads a CSV where each row is one sample.

    Config keys:
      - csv_path: path to CSV/TSV file
      - image_column: column containing image file path (default ``"image_path"``)
      - instruction_column: column containing the prompt/question (default ``"instruction"``)
      - query_type: ``caption``, ``grounding``, ``vqa_binary``, ``vqa_numeric``, ``vqa_semantic``
      - image_root: optional directory prefix for relative image paths
      - delimiter: ``","`` or ``"\t"`` (default: auto-detect)
    """

    def __init__(
        self,
        csv_path: str,
        image_column: str = "image_path",
        instruction_column: str = "instruction",
        query_type: str = "caption",
        image_root: Optional[str] = None,
        delimiter: Optional[str] = None,
        **kwargs,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.csv_path = csv_path
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.query_type = query_type
        self.image_root = image_root or ""
        self.delimiter = delimiter
        self.rows: List[Dict[str, str]] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            # Auto-detect delimiter from header if not provided
            if delimiter is None:
                delimiter = ","
                if "\t" in sample and sample.index("\t") < sample.index(","):
                    delimiter = "\t"
            reader = csv.DictReader(f, delimiter=delimiter)
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = row[self.image_column]
        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        image = Image.open(img_path).convert("RGB")
        instruction = row[self.instruction_column]
        queries = self._build_queries(instruction)
        return {
            "image_id": str(idx),
            "image": image,
            "queries": queries,
            "metadata": {},
        }

    def _build_queries(self, text: str) -> Dict[str, Any]:
        if self.query_type == "caption":
            return {"caption_query": {"instruction": text}}
        if self.query_type == "grounding":
            return {"grounding_query": {"instruction": text}}
        if self.query_type.startswith("vqa_"):
            subtype = self.query_type.replace("vqa_", "")
            return {"attribute_query": {subtype: {"instruction": text}}}
        return {"caption_query": {"instruction": text}}
