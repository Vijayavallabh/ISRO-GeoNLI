"""
End-to-end training script for a VLM with a 2-layer MLP adapter.

- Vision encoder: from config.vision_encoder_name
- Language decoder: from config.language_decoder_name
- Dataset: merged JSON with fields:
    - image_path: str
    - question: str
    - response OR caption: str
    - optional: source (for stratified split)
- Objective: LLaVA-style
    "<image>\nUser: {question}\nAssistant: {answer}</s>"
  with loss only on the assistant answer (including EOS).
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils
import warnings
import random

warnings.filterwarnings("ignore")

# --------- GELUTanh compatibility patch (as before) ---------- #
import transformers.activations as activations
from transformers.activations import NewGELUActivation, GELUActivation
try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh
    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh

# HF / transformers setup  #
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

# 0. CONFIG


@dataclass
class ModelConfig:
    vision_encoder_name: str = "moonshotai/MoonViT-SO-400M"
    language_decoder_name: str = "Qwen/Qwen3-8B"
    freeze_vision_encoder: bool = True
    freeze_language_decoder: bool = True

# -------- Special multimodal tokens --------
IMAGE_TOKEN_INDEX = 151856      # Qwen3 image token index
DEFAULT_IMAGE_TOKEN = "<image>"

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenize text while inserting IMAGE_TOKEN_INDEX wherever <image> appears.
    Matches LLaVA behavior: keeps BOS only once.
    """
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    input_ids = []
    offset = 0
    # If tokenizer added BOS to first chunk, keep only that BOS
    if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])  # preserve a single BOS

    # Insert IMAGE_TOKEN_INDEX between chunks
    for idx, token_chunk in enumerate(chunks):
        # Drop duplicated BOS token from later chunks
        token_chunk = token_chunk[offset:]
        input_ids.extend(token_chunk)

        if idx < len(chunks) - 1:
            # where <image> appeared in prompt
            input_ids.append(image_token_index)

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids



# 1. MLP ADAPTER (token-merging version you pasted)

class MLPAdapter(nn.Module):
    """
    2-layer MLP adapter to project vision features to language decoder space,
    with 2x2 spatial token merging.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.pre_norm = nn.LayerNorm(input_dim)
        self.linear_1 = nn.Linear(input_dim * 4, input_dim * 4)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(input_dim * 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vision features [batch_size, num_tokens, vision_dim]
        Returns:
            Projected features [batch_size, num_tokens/4, language_dim]
        """
        x = self.pre_norm(x)
        B, L, D = x.shape

        # Try to infer spatial H×W from token length
        H = int(L ** 0.5)
        if H * H != L:
            # Possibly CLS + grid
            H = int((L - 1) ** 0.5)
            if H * H == L - 1:
                # Drop CLS
                x = x[:, 1:, :]
                L = L - 1

        W = L // H
        x = x.view(B, H, W, D)                 # [B, H, W, D]

        # 2x2 merging → 4 tokens merged into 1
        x = x.view(B, H // 2, 2, W // 2, 2, D) # [B, H/2, 2, W/2, 2, D]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, -1, D * 4)               # [B, (H/2 * W/2), 4D]

        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)                   # [B, num_merged_tokens, language_dim]
        return x



# 2. VLM WITH ADAPTER (your model.py logic)

class VLMWithAdapter(nn.Module):
    """
    Vision-Language Model with frozen encoders and trainable MLP adapter.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ---- Vision encoder ----
        self.vision_encoder = self._load_vision_encoder()
        if config.freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # ---- Language decoder ----
        # Use AutoModelForCausalLM to ensure .generate is available
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(
            config.language_decoder_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            trust_remote_code=True,
        )
        self.language_decoder = lm
        if config.freeze_language_decoder:
            for p in self.language_decoder.parameters():
                p.requires_grad = False

        # ---- Dimensions ----
        self.vision_dim = self._get_vision_dim()
        self.language_dim = self.language_decoder.config.hidden_size

        # ---- Adapter (only trainable part) ----
        self.adapter = MLPAdapter(
            input_dim=self.vision_dim,
            output_dim=self.language_dim,
        )

        # ---- Processors ----
        self.image_processor = AutoProcessor.from_pretrained(
            config.vision_encoder_name,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.language_decoder_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_vision_encoder(self) -> nn.Module:
        """Load the appropriate vision encoder based on config."""
        # For MoonViT-like models that have .vision_tower()
        base = AutoModel.from_pretrained(
            self.config.vision_encoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        )
        if hasattr(base, "vision_tower"):
            return base.vision_tower()
        return base  # fallback

    def _get_vision_dim(self) -> int:
        """Get vision encoder output dimension."""
        cfg = self.vision_encoder.config
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        if hasattr(cfg, "projection_dim"):
            return cfg.projection_dim
        raise ValueError("Cannot determine vision encoder output dimension")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.

        Args:
            pixel_values: [batch_size, C, H, W]
        Returns:
            [batch_size, num_tokens, vision_dim]
        """
        if self.config.freeze_vision_encoder:
            ctx = torch.no_grad()
        else:
            ctx = torch.enable_grad()

        with ctx:
            outputs = self.vision_encoder(pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                vision_features = outputs.last_hidden_state
            elif hasattr(outputs, "pooler_output"):
                vision_features = outputs.pooler_output.unsqueeze(1)
            else:
                raise ValueError("Cannot extract vision features from encoder")

        return vision_features

    def _prepare_labels(self, labels: torch.Tensor, num_vision_tokens: int) -> torch.Tensor:
        """
        Prepend -100 for vision tokens to ignore them in loss.
        """
        B = labels.shape[0]
        vision_labels = torch.full(
            (B, num_vision_tokens),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([vision_labels, labels], dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B = pixel_values.shape[0]

        # 1) Encode images
        vision_features = self.encode_image(pixel_values)   # [B, Lv, Dv]

        # 2) Project through adapter (trainable)
        projected_features = self.adapter(vision_features)  # [B, Lm, Dl]

        # 3) Text embeddings
        text_embeds = self.language_decoder.get_input_embeddings()(input_ids)  # [B, Lt, Dl]

        # 4) Combine vision + text
        num_vision_tokens = projected_features.shape[1]
        combined_embeds = torch.cat([projected_features, text_embeds], dim=1)  # [B, Lm+Lt, Dl]

        # 5) Attention mask
        vision_attention_mask = torch.ones(
            B, num_vision_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention_mask = torch.cat(
            [vision_attention_mask, attention_mask], dim=1
        )

        # 6) Prepare labels (ignore vision tokens)
        dec_labels = None
        if labels is not None:
            dec_labels = self._prepare_labels(labels, num_vision_tokens)

        outputs = self.language_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=dec_labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Image + text-conditioned generation.
        """
        B = pixel_values.shape[0]

        vision_features = self.encode_image(pixel_values)
        projected_features = self.adapter(vision_features)
        text_embeds = self.language_decoder.get_input_embeddings()(input_ids)

        num_vision_tokens = projected_features.shape[1]
        combined_embeds = torch.cat([projected_features, text_embeds], dim=1)

        vision_attention_mask = torch.ones(
            B, num_vision_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention_mask = torch.cat(
            [vision_attention_mask, attention_mask], dim=1
        )

        return self.language_decoder.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

    def gradient_checkpointing_enable(self):
        if hasattr(self.language_decoder, "gradient_checkpointing_enable"):
            self.language_decoder.gradient_checkpointing_enable()

    def get_trainable_parameters(self):
        """Return only the adapter parameters for optimization (for manual training)."""
        return self.adapter.parameters()

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )



# 3. DATASET 


class UnifiedDataset(Dataset):
    """
    Works with LLaVA-style JSON:
      {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "... <image>"},
            {"from": "gpt",   "value": "answer text"}
        ]
      }
    """
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        img = Image.open(x["image"]).convert("RGB")

        conv = x["conversations"]
        human_msg = conv[0]["value"]       
        gpt_msg   = conv[1]["value"]

        return {
            "image": img,
            "human": human_msg,
            "gpt": gpt_msg
        }


def stratified_split(json_path: str, train_ratio: float = 0.8, seed: int = 42):
    with open(json_path, "r") as f:
        data = json.load(f)

    sources = [x.get("source", "Unknown") for x in data]
    idx = list(range(len(data)))

    train_idx, eval_idx = train_test_split(
        idx,
        train_size=train_ratio,
        random_state=seed,
        stratify=sources,
    )
    return train_idx, eval_idx, data



# 4. COLLATOR (EOS + masking fix)


class CustomDataCollator:
    """
    Input conversation example:
        human: "Render a summary...\n<image>"
        gpt:   "This is a ..."
    """

    def __init__(self, tokenizer, image_processor, max_length=256):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        human_msgs = [b["human"] for b in batch]
        gpt_msgs   = [b["gpt"] for b in batch]

        texts = []
        prefixes = []

        for h, a in zip(human_msgs, gpt_msgs):
            # h already includes <image>
            prefix = f"User: {h}\nAssistant: "
            full_text = prefix + a + self.tokenizer.eos_token

            prefixes.append(prefix)
            texts.append(full_text)

        # Tokenize with LLaVA multimodal tokenizer
        input_ids = [
            tokenizer_image_token(t, self.tokenizer, return_tensors="pt")
            for t in texts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        # Mask everything before the Assistant answer
        for i, prefix in enumerate(prefixes):
            prefix_ids = tokenizer_image_token(prefix, self.tokenizer)
            labels[i, :len(prefix_ids)] = -100

        # Process images
        imgs = self.image_processor(images=images, return_tensors="pt")

        return {
            "pixel_values": imgs.pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }




# 5. TRAINING LOOP


DATA_PATH = "/home/LLaVA-Pretrain/blip_laion_cc_sbu_558k_fixed.json"
print(f"Loading dataset once from {DATA_PATH} ...")

with open(DATA_PATH, "r") as f:
    GLOBAL_DATA = json.load(f)

print(f"Loaded {len(GLOBAL_DATA):,} samples.")

def train():
  
    # 1) SAMPLE FIRST 100K (or random 100K)
    MAX_SAMPLES = 100_000
    print(f"📦 Sampling {MAX_SAMPLES:,} examples ...")

    if len(GLOBAL_DATA) >= MAX_SAMPLES:
        sampled_data = random.sample(GLOBAL_DATA, MAX_SAMPLES)
    else:
        sampled_data = GLOBAL_DATA

    # 2) Save to temporary file (DDP-safe)

    TEMP_JSON = "/home/temp_pretrain_sampled_100k.json"
    print(f"💾 Writing temp dataset → {TEMP_JSON}")

    with open(TEMP_JSON, "w") as f:
        json.dump(sampled_data, f)


    # 3) Run stratified split and dataset creation

    train_idx, eval_idx, _ = stratified_split(TEMP_JSON)

    full_dataset = UnifiedDataset(TEMP_JSON)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    eval_dataset = torch.utils.data.Subset(full_dataset, eval_idx)

    cfg = ModelConfig() 

    # Build model
    model = VLMWithAdapter(cfg)
    model.print_trainable_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tokenizer & image processor from model (to stay consistent)
    tokenizer = model.tokenizer
    image_processor = model.image_processor

 
    collator = CustomDataCollator(tokenizer, image_processor, max_length=256)

    args = TrainingArguments(
        output_dir="./checkpoints/vlm_adapter_pretrain",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-3,
        bf16=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        save_safetensors=False,
        logging_steps=20,
        log_level="info",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=False,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    torch.save(model.adapter.state_dict(), "vlm_mlp_adapter_pretrained.bin")
    print("MLP adapter pretraining complete.")


if __name__ == "__main__":
    train()
