from __future__ import annotations
import unsloth
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import IterableDataset, load_dataset
from PIL import Image
import evaluate

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

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
    """
    Parse a raw VRSBench example into a structured VRSSample.
    Assumes fields:
      - image: relative path
      - caption: global caption
      - objects: list with 'obj_corner' and 'referring_sentence'
      - qa_pairs: list with 'question' and 'answer'
    """
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


class VRSStreamingSplit(IterableDataset):
    """
    Streaming wrapper over a HuggingFace streaming split, yielding VRSSample.
    Subclasses datasets.IterableDataset so Unsloth skips len() checks.
    """
    def __init__(self, dataset: IterableDataset, image_dir: str, split_type: str = "all"):
        self.dataset: IterableDataset = dataset
        self.image_root = Path(image_dir)
        self.split_type = split_type

    def set_epoch(self, epoch: int):
        """
        Required by Trainer/Accelerate for distributed training or epoch tracking.
        """
        self._epoch = epoch
        # If the underlying HF dataset supports set_epoch, pass it down.
        # HF streaming datasets usually handle shuffling via .shuffle(seed=...) 
        # rather than explicit set_epoch, but having this method prevents the AttributeError.
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
            
    def __iter__(self) -> Iterator[VRSSample]:
        for example in self.dataset:
            if self.split_type == "all":
                yield _parse_example(example, self.image_root)
            else:
                # Optional hash-based split if you want train/val from one stream
                hash_val = int(hashlib.md5(example["image"].encode()).hexdigest(), 16)
                remainder = hash_val % 5
                if (self.split_type == "train" and remainder != 0) or (
                    self.split_type == "val" and remainder == 0
                ):
                    yield _parse_example(example, self.image_root)


def build_vrs_dataloaders_train_test_only() -> Dict[str, VRSStreamingSplit]:
    """
    Function that provides only train and test splits without any validation split.

    If you are using the HF version, the dataset name is often "xiang709/VRSBench".
    Replace "VRSBench" below with that if needed.
    """
    train_ds = load_dataset("VRSBench", split="train", streaming=True)
    test_ds = load_dataset("VRSBench", split="validation", streaming=True)

    return {
        "train": VRSStreamingSplit(train_ds, "VRSBench/Images_train", split_type="all"),
        "test": VRSStreamingSplit(test_ds, "VRSBench/Images_val", split_type="all"),
    }


# =========================
# 2. Adapter to Unsloth vision chat format (caption + VQA)
# =========================

CAPTION_INSTRUCTION = (
    "You are a remote sensing expert. "
    "Describe this aerial or satellite image in detail for an analyst."
)

VQA_SYSTEM_PREFIX = (
    "You are a remote sensing expert. Answer the question given the image."
)


def make_caption_messages(sample: VRSSample) -> Dict:
    """
    Single-turn captioning conversation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_INSTRUCTION},
                {"type": "image", "image": sample.image},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample.caption},
            ],
        },
    ]
    return {"messages": messages}


def make_vqa_messages(
    sample: VRSSample, question: str, answer: str
) -> Dict:
    """
    Single-turn VQA conversation for one (question, answer) pair.
    """
    user_text = f"{VQA_SYSTEM_PREFIX} Question: {question}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]
    return {"messages": messages}


class VRSMessagesDataset(IterableDataset):
    """
    HF IterableDataset compatible with Unsloth/Trainer, 
    implementing set_epoch to satisfy Accelerate/Trainer requirements.
    """
    def __init__(
        self,
        vrs_split: VRSStreamingSplit,
        include_caption: bool = True,
        include_vqa: bool = True,
    ):
        self.vrs_split = vrs_split
        self.include_caption = include_caption
        self.include_vqa = include_vqa
        # Initialize _epoch for Trainer compatibility
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """
        Called by Trainer at the beginning of each epoch.
        """
        self._epoch = epoch
        if hasattr(self.vrs_split, "set_epoch"):
             self.vrs_split.set_epoch(epoch)

    def __iter__(self):
        for sample in self.vrs_split:
            # Captioning example
            if self.include_caption and sample.caption:
                yield make_caption_messages(sample)

            # VQA examples
            if self.include_vqa and sample.vqa_questions and sample.vqa_answers:
                n = len(sample.vqa_questions)
                for q, a in zip(sample.vqa_questions[:n], sample.vqa_answers[:n]):
                    if not q or not a:
                        continue
                    yield make_vqa_messages(sample, q, a)

# =========================
# 3. Load Qwen3-VL-8B via Unsloth and configure LoRA (LM attention + MLP only)
# =========================

model_name = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,                  # QLoRA in 4-bit
    use_gradient_checkpointing="unsloth",
)

# Attach LoRA only to language_model attention + MLP layers
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,       # keep vision tower frozen
    finetune_language_layers=True,      # only language_model gets adapters
    finetune_attention_modules=True,    # LoRA on attention projections
    finetune_mlp_modules=False,          # LoRA on FFN/MLP
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,

)

FastVisionModel.for_training(model)     # enable training mode


# =========================
# 4. Build streaming datasets (caption + VQA) for Unsloth
# =========================

splits = build_vrs_dataloaders_train_test_only()

# Multi-task dataset: caption #+ VQA pairs per image
train_dataset = VRSMessagesDataset(
    splits["train"],
    include_caption=True,
    include_vqa=False,
)
eval_dataset = VRSMessagesDataset(
    splits["test"],
    include_caption=True,
    include_vqa=False,
)

data_collator = UnslothVisionDataCollator(model, tokenizer)

training_args = SFTConfig(
    max_seq_length=2048,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=5000,                    # better for streaming than num_train_epochs
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    output_dir="qwen3vl_vrsbench_caption",
    report_to="none",
        bf16=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,       # HF IterableDataset -> no len() issue
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    args=training_args,
)
trainer.train()
