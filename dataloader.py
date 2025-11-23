from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from datasets import IterableDataset, load_dataset
from PIL import Image
import hashlib

@dataclass
class VRSSample:
    image: Image.Image
    caption: str
    bbox: List[List[int]]
    bbox_questions: List[str]
    vqa_questions: List[str]
    vqa_answers: List[str]


def _open_image(root: Path, rel_path: str) -> Image.Image:
    return Image.open(root / rel_path).convert("RGB")


def _parse_example(example: Dict, image_root: Path) -> VRSSample:
    # if "objects" not in example:
    #     print(f"Missing 'objects' in example: {example}")
    #     return None  # Skip if missing
    bbox = [obj["obj_corner"] for obj in example["objects"]]
    bbox_questions = [
        f"Give me the location of the {obj['referring_sentence']}"
        for obj in example["objects"]
    ]
    vqa_questions = [pair["question"] for pair in example["qa_pairs"]]
    vqa_answers = [pair["answer"] for pair in example["qa_pairs"]]
    return VRSSample(
        image=_open_image(image_root, example["image"]),
        caption=example["caption"],
        bbox=bbox,
        bbox_questions=bbox_questions,
        vqa_questions=vqa_questions,
        vqa_answers=vqa_answers,
    )


class VRSStreamingSplit(Iterable[VRSSample]):
    def __init__(self, dataset: IterableDataset, image_dir: str, split_type: str = "all"):
        self.dataset: IterableDataset = dataset
        self.image_root = Path(image_dir)
        self.split_type = split_type

    def __iter__(self) -> Iterator[VRSSample]:
        for example in self.dataset:
            if self.split_type == "all":
                yield _parse_example(example, self.image_root)
            else:
                hash_val = int(hashlib.md5(example["image"].encode()).hexdigest(), 16)
                remainder = hash_val % 5
                if (self.split_type == "train" and remainder != 0) or (self.split_type == "val" and remainder == 0):
                    yield _parse_example(example, self.image_root)


def build_vrs_dataloaders() -> Dict[str, VRSStreamingSplit]:
    train_ds = load_dataset("VRSBench", split="train", streaming=True)
    test_ds = load_dataset("VRSBench", split="validation", streaming=True)
    
    return {
        "train": VRSStreamingSplit(train_ds, "VRSBench/Images_train", split_type="train"),
        "val": VRSStreamingSplit(train_ds, "VRSBench/Images_train", split_type="val"),
        "test": VRSStreamingSplit(test_ds, "VRSBench/Images_val", split_type="all"),
    }

def build_vrs_dataloaders_train_test_only() -> Dict[str, VRSStreamingSplit]:
    """
    Function that provides only train and test splits without any validation split.
    """
    train_ds = load_dataset("VRSBench", split="train", streaming=True)
    test_ds = load_dataset("VRSBench", split="validation", streaming=True)
    
    return {
        "train": VRSStreamingSplit(train_ds, "VRSBench/Images_train", split_type="all"),
        "test": VRSStreamingSplit(test_ds, "VRSBench/Images_val", split_type="all"),
    }