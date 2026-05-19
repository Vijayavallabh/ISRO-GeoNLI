"""
Image-folder dataset with JSON sidecars.
Each image file ``img.jpg`` can have a matching ``img.json`` file in the same
folder (or in a parallel annotations folder) containing the queries.
"""

import json
import os
from typing import Any, Dict, List, Optional
from PIL import Image

from geonli.core.base import GeoNLIDataset
from geonli.core.registry import register_dataset


@register_dataset("image_folder")
class ImageFolderGeoNLIDataset(GeoNLIDataset):
    """
    Loads images from a folder and reads GeoNLI-format JSON sidecars.

    Config keys:
      - image_dir: folder containing images
      - annotation_dir: optional separate folder with JSON files (default = image_dir)
      - extensions: list of valid image extensions (default ``[".jpg", ".jpeg", ".png"]``)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        **kwargs,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir or image_dir
        self.extensions = extensions or [".jpg", ".jpeg", ".png"]
        self.samples: List[str] = []

        for fname in sorted(os.listdir(image_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.extensions:
                self.samples.append(fname)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.samples[idx]
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(self.image_dir, fname)
        ann_path = os.path.join(self.annotation_dir, f"{base}.json")

        image = Image.open(img_path).convert("RGB")

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                data = json.load(f)
            queries = data.get("queries", {})
            metadata = data.get("metadata", {})
        else:
            queries = {}
            metadata = {}

        return {
            "image_id": base,
            "image": image,
            "queries": queries,
            "metadata": metadata,
        }
