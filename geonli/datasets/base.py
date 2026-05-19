"""
Dataset implementations that emit the standard GeoNLI dict format.
"""

import json
import os
from typing import Any, Dict, List, Optional
from PIL import Image

try:
    from torch.utils.data import Dataset
except Exception:
    # If PyTorch is not installed, provide a minimal shim
    class Dataset:
        pass

from geonli.core.base import GeoNLIDataset
from geonli.core.registry import register_dataset


@register_dataset("json_dataset")
class JsonGeoNLIDataset(GeoNLIDataset, Dataset):
    """
    Loads samples from a GeoNLI-style JSON file or directory of JSON files.
    Expected dict format (per sample):
        {
            "input_image": {
                "image_id": "...",
                "image_url" | "image_path" | "image_base64": "...",
                "metadata": {...}
            },
            "queries": {...}
        }
    """

    def __init__(
        self,
        input_json: str,
        image_root: Optional[str] = None,
        **kwargs,
    ):
        self.samples: List[Dict[str, Any]] = []
        self.image_root = image_root or ""

        if os.path.isfile(input_json):
            with open(input_json, "r") as f:
                data = json.load(f)
            # Support both a single object and a list of objects
            if isinstance(data, list):
                self.samples = data
            else:
                self.samples = [data]
        elif os.path.isdir(input_json):
            for fname in sorted(os.listdir(input_json)):
                if fname.endswith(".json"):
                    with open(os.path.join(input_json, fname), "r") as f:
                        self.samples.append(json.load(f))
        else:
            raise FileNotFoundError(f"input_json not found: {input_json}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_info = sample.get("input_image", {})
        image = self._load_image(image_info)
        return {
            "image_id": image_info.get("image_id", f"sample_{idx}"),
            "image": image,
            "queries": sample.get("queries", {}),
            "metadata": image_info.get("metadata", {}),
        }

    def _load_image(self, image_info: Dict[str, Any]) -> Image.Image:
        import base64
        from io import BytesIO

        # 1. Local path
        path = image_info.get("image_path")
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB")

        # 2. Relative to image_root
        img_id = image_info.get("image_id", "")
        if self.image_root and img_id:
            candidate = os.path.join(self.image_root, img_id)
            if os.path.exists(candidate):
                return Image.open(candidate).convert("RGB")

        # 3. Base64
        b64 = image_info.get("image_base64", "")
        if b64:
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            return Image.open(BytesIO(data)).convert("RGB")

        # 4. URL
        url = image_info.get("image_url", "")
        if url:
            import requests
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")

        raise ValueError("No valid image source found in input_image.")
