"""
Modular LoRA Fine-tuning Script for Qwen3-VL-8B-Instruct
Optimized for Qwen3-VL architecture (ViT-based vision + Qwen3 text model)
"""
from dataloader import build_vrs_dataloaders_train_test_only
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from PIL import Image

# ==================== Configuration ====================

@dataclass
class ModelConfig:
    """Configuration for the base VLM model"""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Generate quantization config if needed"""
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.torch_dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning with Qwen3-VL specific architecture support"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM
    
    # Target modules configuration
    target_modules: Optional[List[str]] = None
    
    # Preset options for Qwen3-VL architecture
    target_preset: Literal[
        "language_attention",           # Only LLM attention layers
        "language_mlp",                 # Only LLM MLP layers
        "language_all",                 # All LLM components
        "vision_attention",             # Only vision encoder attention
        "vision_mlp",                   # Only vision encoder MLP
        "vision_all",                   # All vision encoder components
        "merger",                       # Multi-modal projector/merger only
        "attention_all",                # All attention layers (vision + language)
        "mlp_all",                      # All MLP layers (vision + language)
        "full_model",                   # Everything except embeddings/lm_head
        "custom"                        # Use custom_modules
    ] = "language_attention"
    
    # Component selection for fine-grained control
    include_vision_encoder: bool = False
    include_merger: bool = False
    include_language_model: bool = True
    
    # Custom module names (used when target_preset="custom")
    custom_modules: List[str] = field(default_factory=list)
    
    # Module pattern exclusions (optional)
    modules_to_exclude: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    
    def get_target_modules(self, model) -> List[str]:
        """
        Determine target modules based on Qwen3-VL architecture
        
        Model structure:
        - model.visual.blocks[*].attn.{qkv, proj}
        - model.visual.blocks[*].mlp.{linear_fc1, linear_fc2}
        - model.visual.merger.{linear_fc1, linear_fc2}
        - model.visual.deepstack_merger_list[*].{linear_fc1, linear_fc2}
        - model.language_model.layers[*].self_attn.{q_proj, k_proj, v_proj, o_proj}
        - model.language_model.layers[*].mlp.{gate_proj, up_proj, down_proj}
        """
        if self.target_modules is not None:
            return self.target_modules
            
        if self.target_preset == "custom":
            return self.custom_modules
        
        # Define module groups for Qwen3-VL
        vision_attention = ["qkv", "proj"]  # Vision encoder attention
        vision_mlp = ["linear_fc1", "linear_fc2"]  # Vision encoder MLP
        
        merger_modules = ["linear_fc1", "linear_fc2"]  # Projector/merger
        
        # Qwen3 text model uses standard transformer attention
        language_attention = [
            "q_proj",    # Query projection
            "k_proj",    # Key projection
            "v_proj",    # Value projection
            "o_proj"     # Output projection
        ]
        
        language_mlp = ["gate_proj", "up_proj", "down_proj"]  # Standard MLP
        
        # Build target list based on preset
        targets = []
        
        if self.target_preset == "language_attention":
            targets = language_attention
            
        elif self.target_preset == "language_mlp":
            targets = language_mlp
            
        elif self.target_preset == "language_all":
            targets = language_attention + language_mlp
            
        elif self.target_preset == "vision_attention":
            targets = vision_attention
            
        elif self.target_preset == "vision_mlp":
            targets = vision_mlp
            
        elif self.target_preset == "vision_all":
            targets = vision_attention + vision_mlp
            
        elif self.target_preset == "merger":
            targets = merger_modules
            
        elif self.target_preset == "attention_all":
            targets = vision_attention + language_attention
            
        elif self.target_preset == "mlp_all":
            targets = vision_mlp + language_mlp
            
        elif self.target_preset == "full_model":
            targets = (vision_attention + vision_mlp + 
                      merger_modules + 
                      language_attention + language_mlp)
        
        # Apply component filters
        if not self.include_vision_encoder:
            targets = [t for t in targets if t not in vision_attention + vision_mlp]
        
        if not self.include_merger:
            targets = [t for t in targets if t not in merger_modules]
            
        if not self.include_language_model:
            targets = [t for t in targets if t not in language_attention + language_mlp]
        
        return targets
    
    def get_modules_to_save(self) -> Optional[List[str]]:
        """
        Specify additional modules to save during training
        Useful for merger or embedding layers
        """
        modules = []
        
        if self.include_merger:
            modules.extend(["visual.merger", "visual.deepstack_merger_list"])
            
        return modules if modules else None
    
    def to_peft_config(self, model) -> LoraConfig:
        """Convert to PEFT LoraConfig object"""
        target_modules = self.get_target_modules(model)
        modules_to_save = self.get_modules_to_save()
        
        print(f"\n{'='*60}")
        print(f"LoRA Configuration for Qwen3-VL:")
        print(f"  Rank: {self.r}, Alpha: {self.lora_alpha}, Dropout: {self.lora_dropout}")
        print(f"  Target preset: {self.target_preset}")
        print(f"  Target modules: {target_modules}")
        print(f"  Vision Encoder: {self.include_vision_encoder}")
        print(f"  Merger/Projector: {self.include_merger}")
        print(f"  Language Model: {self.include_language_model}")
        if modules_to_save:
            print(f"  Additional modules to save: {modules_to_save}")
        print(f"{'='*60}\n")
        
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
    
    @staticmethod
    def inspect_model_modules(model):
        """
        Utility function to inspect model structure and find module names
        Useful for debugging and finding target modules
        """
        print("\n" + "="*60)
        print("Qwen3-VL Model Module Analysis")
        print("="*60 + "\n")
        
        vision_modules = set()
        merger_modules = set()
        language_modules = set()
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]
                
                if 'visual' in name and 'merger' not in name:
                    vision_modules.add(module_name)
                elif 'merger' in name:
                    merger_modules.add(module_name)
                elif 'language_model' in name:
                    language_modules.add(module_name)
        
        print("Vision Encoder Linear Modules:")
        print(f"  {sorted(vision_modules)}\n")
        
        print("Merger/Projector Linear Modules:")
        print(f"  {sorted(merger_modules)}\n")
        
        print("Language Model Linear Modules:")
        print(f"  {sorted(language_modules)}\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        vision_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'visual' in n and 'merger' not in n)
        merger_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'merger' in n)
        language_params = sum(p.numel() for n, p in model.named_parameters() 
                             if 'language_model' in n)
        
        print("Parameter Distribution:")
        print(f"  Total: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Vision: {vision_params:,} ({vision_params/total_params*100:.1f}%)")
        print(f"  Merger: {merger_params:,} ({merger_params/total_params*100:.1f}%)")
        print(f"  Language: {language_params:,} ({language_params/total_params*100:.1f}%)")
        print("="*60 + "\n")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing"""
    dataset_name: str = "VRSBench"
    task: Literal["vqa", "caption", "both"] = "both"
    streaming: bool = False
    image_column: str = "image"
    
    # Task-specific column names
    vqa_question_column: str = "question"
    vqa_answer_column: str = "answer"
    caption_column: str = "caption"
    
    # Data splits
    train_split: str = "train"
    eval_split: str = "test"
    
    # Processing
    max_length: int = 2048


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    output_dir: str = "./vlm_lora_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            optim=self.optim,
            max_grad_norm=self.max_grad_norm,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )


@dataclass
class VLMDataCollator:
    """
    Custom data collator for Qwen3-VL that handles flattening of pixel_values
    and other image-related tensors.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        first = features[0]
        
        # Debug: Print first feature shapes
        # print(f"First feature shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in first.items()]}")
        
        # Keys that need special handling for Qwen3-VL
        pixel_keys = ["pixel_values"]
        grid_keys = ["image_grid_thw"]
        
        for k, v in first.items():
            if k in pixel_keys:
                # Concatenate pixel values along first dimension (flatten all image patches)
                # Each feature[k] has shape [num_patches_per_image, C, H, W]
                # Result: [total_patches_in_batch, C, H, W]
                batch[k] = torch.cat([f[k] for f in features], dim=0)
                
            elif k in grid_keys:
                # Concatenate grid dimensions (each sample has [num_images_in_sample, 3])
                # Result: [total_images_in_batch, 3]
                batch[k] = torch.cat([f[k] for f in features], dim=0)
                
            else:
                # Default stacking for text and other tensors
                try:
                    batch[k] = torch.stack([f[k] for f in features])
                except Exception as e:
                    # Fallback for non-tensor data or incompatible shapes
                    batch[k] = [f[k] for f in features]
        
        # Debug: Print batch shapes
        # print(f"Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()]}")
        
        return batch

# ==================== Dataset Implementation ====================

class StreamingVLMDataset(Dataset):
    """
    Dataset class for handling streaming VRSBench data from dataloader.py
    Flattens samples to handle multiple VQA pairs per image
    """
    
    def __init__(self, iterable, processor, dataset_config: DatasetConfig):
        self.samples = list(iterable)  # Collect samples from the iterable
        self.processor = processor
        self.config = dataset_config
        
        # Flatten samples for VQA and caption tasks
        self.flattened = []
        for sample in self.samples:
            # Handle caption task
            if self.config.task in ["caption", "both"]:
                self.flattened.append({
                    "image": sample.image,
                    "text": sample.caption,
                    "task": "caption"
                })
            # Handle VQA task (multiple questions per sample)
            if self.config.task in ["vqa", "both"]:
                for q, a in zip(sample.vqa_questions, sample.vqa_answers):
                    self.flattened.append({
                        "image": sample.image,
                        "question": q,
                        "answer": a,
                        "task": "vqa"
                    })
        
    
    def __len__(self) -> int:
        return len(self.flattened)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flattened[idx]
        image = item["image"]
        task = item["task"]
        
        if task == "vqa":
            question = item["question"]
            answer = item["answer"]
            # Qwen3-VL uses <|vision_start|><|image_pad|><|vision_end|> for images
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>\nQuestion: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"
        elif task == "caption":
            caption = item["text"]
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>\nDescribe the contents of the image in detail:"
            full_text = f"{prompt} {caption}"     
        
        # Process with the VLM processor
        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # For Qwen3-VL, don't squeeze all dimensions blindly
        # Keep image_grid_thw with shape [1, 3] for single image
        # pixel_values should be [num_patches, channels, height, width]
        encoding_processed = {}
        for k, v in encoding.items():
            if k in ["image_grid_thw"]:
                # Keep shape [1, 3] for grid dimensions
                encoding_processed[k] = v if v.dim() > 1 else v.unsqueeze(0)
            elif k in ["pixel_values"]:
                # Remove only the batch dimension (first dim), keep the rest
                encoding_processed[k] = v.squeeze(0)
            else:
                # Remove batch dimension for text tensors
                encoding_processed[k] = v.squeeze(0)
        
        # Create labels (same as input_ids for causal LM, with prompt masked)
        labels = encoding_processed["input_ids"].clone()
        
        # Mask the prompt tokens (only train on the answer/caption)
        prompt_encoding = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_length] = -100  # Ignore prompt in loss
        
        # Mask padding tokens in labels
        if "attention_mask" in encoding_processed:
            labels[encoding_processed["attention_mask"] == 0] = -100
        
        encoding_processed["labels"] = labels
        
        return encoding_processed

class VLMDataset(Dataset):
    """
    Generic dataset class for Vision-Language Models
    Handles both VQA and captioning tasks
    """
    
    def __init__(
        self,
        dataset,
        processor,
        dataset_config: DatasetConfig,
        split: str = "train"
    ):
        self.dataset = dataset
        self.processor = processor
        self.config = dataset_config
        self.split = split
        
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flattened[idx]
        image = item["image"]
        task = item["task"]
        
        if task == "vqa":
            question = item["question"]
            answer = item["answer"]
            # Qwen3-VL uses <|vision_start|><|image_pad|><|vision_end|> for images
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>\nQuestion: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"
        elif task == "caption":
            caption = item["text"]
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>\nDescribe the contents of the image in detail:"
            full_text = f"{prompt} {caption}"     
        
        # Process with the VLM processor
        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # Debug: Print shapes before processing
        # print(f"Raw encoding shapes: {[(k, v.shape) for k, v in encoding.items()]}")
        
        # For Qwen3-VL, handle dimensions carefully
        encoding_processed = {}
        for k, v in encoding.items():
            if k == "image_grid_thw":
                # image_grid_thw should be [num_images, 3] where each row is [T, H, W]
                # After processor it's [batch_size=1, num_images, 3]
                # We need to squeeze the batch dimension but keep [num_images, 3]
                if v.dim() == 3:  # [1, num_images, 3]
                    encoding_processed[k] = v.squeeze(0)  # -> [num_images, 3]
                elif v.dim() == 2:  # Already [num_images, 3]
                    encoding_processed[k] = v
                else:
                    raise ValueError(f"Unexpected image_grid_thw shape: {v.shape}")
                
            elif k == "pixel_values":
                # pixel_values should be [num_patches, C, H, W]
                # After processor it's [batch_size=1, num_patches, C, H, W]
                if v.dim() == 5:  # [1, num_patches, C, H, W]
                    encoding_processed[k] = v.squeeze(0)  # -> [num_patches, C, H, W]
                elif v.dim() == 4:  # Already [num_patches, C, H, W]
                    encoding_processed[k] = v
                else:
                    raise ValueError(f"Unexpected pixel_values shape: {v.shape}")
                    
            else:
                # For text tensors, squeeze batch dimension
                if v.dim() > 1:
                    encoding_processed[k] = v.squeeze(0)
                else:
                    encoding_processed[k] = v
        
        # Create labels (same as input_ids for causal LM, with prompt masked)
        labels = encoding_processed["input_ids"].clone()
        
        # Mask the prompt tokens (only train on the answer/caption)
        prompt_encoding = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_length] = -100  # Ignore prompt in loss
        
        # Mask padding tokens in labels
        if "attention_mask" in encoding_processed:
            labels[encoding_processed["attention_mask"] == 0] = -100
        
        encoding_processed["labels"] = labels
        
        return encoding_processed


# ==================== Model Setup ====================

def load_model_and_processor(model_config: ModelConfig):
    """
    Load Qwen3-VL model and processor with optional quantization
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_config.model_name}")
    print(f"{'='*60}\n")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )
    
    # Setup quantization
    quantization_config = model_config.get_quantization_config()
    
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_config.model_name,
        torch_dtype=getattr(torch, model_config.torch_dtype),
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        quantization_config=quantization_config,
    )

    if model_config.load_in_4bit or model_config.load_in_8bit:
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=False  
        )
        
        # DON'T manually enable gradient checkpointing - it causes issues with quantized models
        print("Note: Gradient checkpointing disabled for quantized training")
    
    # Enable gradient checkpointing for inputs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Manual implementation if method doesn't exist
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        print("  ✓ Manually enabled input gradients")
    
    # Disable use_cache to avoid conflicts with gradient checkpointing
    model.config.use_cache = False
    
    # Ensure model and tokenizer are in sync with special tokens
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    
    # Qwen3-VL special tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None
    model.config.eos_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer Special Tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")
    print(f"Model config: use_cache={model.config.use_cache}")
    print(f"\nModel loaded successfully!\n")
    
    return model, processor


def apply_lora(model, lora_config: LoRAConfig):
    """
    Apply LoRA adapters to the Qwen3-VL model
    """
    print("\n" + "="*60)
    print("Applying LoRA to Qwen3-VL Model")
    print("="*60 + "\n")
    
    # Optional: Inspect model structure first
    # Uncomment to see detailed module breakdown
    # lora_config.inspect_model_modules(model)
    
    # Get PEFT config
    peft_config = lora_config.to_peft_config(model)
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters breakdown
    print("="*60)
    print("Trainable Parameters Summary")
    print("="*60)
    model.print_trainable_parameters()
    
    # Additional breakdown by component
    vision_trainable = sum(p.numel() for n, p in model.named_parameters() 
                          if p.requires_grad and 'visual' in n and 'merger' not in n)
    merger_trainable = sum(p.numel() for n, p in model.named_parameters() 
                          if p.requires_grad and 'merger' in n)
    language_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if p.requires_grad and 'language_model' in n)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if total_trainable > 0:
        print("\nComponent Breakdown:")
        if vision_trainable > 0:
            print(f"  Vision Encoder: {vision_trainable:,} "
                  f"({vision_trainable/total_trainable*100:.2f}%)")
        if merger_trainable > 0:
            print(f"  Merger/Projector: {merger_trainable:,} "
                  f"({merger_trainable/total_trainable*100:.2f}%)")
        if language_trainable > 0:
            print(f"  Language Model: {language_trainable:,} "
                  f"({language_trainable/total_trainable*100:.2f}%)")
    
    print("="*60 + "\n")
    
    return model


# ==================== Data Loading ====================

def load_training_data(dataset_config: DatasetConfig, processor):
    """
    Load and prepare training and evaluation datasets
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_config.dataset_name}")
    print(f"{'='*60}\n")
    
    if "VRSBench" in dataset_config.dataset_name:
        # Use the custom dataloader from dataloader.py
        print("Using custom VRSBench dataloader...")
        dataloaders = build_vrs_dataloaders_train_test_only()
        train_iterable = dataloaders["train"]
        eval_iterable = dataloaders["test"]
        
        train_dataset = StreamingVLMDataset(train_iterable, processor, dataset_config)
        eval_dataset = StreamingVLMDataset(eval_iterable, processor, dataset_config)
    else:
        # Original loading for other datasets
        print(f"Loading from HuggingFace: {dataset_config.dataset_name}")
        dataset = load_dataset(
            dataset_config.dataset_name,
            streaming=dataset_config.streaming
        )
        
        # Create dataset wrappers
        DatasetClass = VLMDataset
        
        train_dataset = DatasetClass(
            dataset[dataset_config.train_split],
            processor,
            dataset_config,
            split="train"
        )
        
        eval_dataset = DatasetClass(
            dataset[dataset_config.eval_split],
            processor,
            dataset_config,
            split="eval"
        ) if dataset_config.eval_split in dataset else None
    
    print(f"✓ Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"✓ Eval samples: {len(eval_dataset)}")
    print()
    
    return train_dataset, eval_dataset


# ==================== Training ====================

def train_vlm(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
):
    """
    Main training function that orchestrates the entire pipeline
    """
    
    # Load model and processor
    model, processor = load_model_and_processor(model_config)
    
    # Apply LoRA
    model = apply_lora(model, lora_config)
    
    # Load datasets
    train_dataset, eval_dataset = load_training_data(dataset_config, processor)
    
    # Setup training arguments
    training_args = training_config.to_training_args()

    data_collator = VLMDataCollator()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
    )
    
    # Train
    print("="*60)
    print("Starting Training")
    print("="*60 + "\n")
    trainer.train()
    
    # Save final model
    final_model_path = Path(training_config.output_dir) / "final_model"
    trainer.save_model(final_model_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {final_model_path}")
    print(f"{'='*60}\n")
    
    return model, processor, trainer


# ==================== Inference ====================

def inference_example(model, processor, image_path: str, question: str = None):
    """
    Example inference function
    """
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare prompt
    if question:
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>\nQuestion: {question}\nAnswer:"
    else:
        prompt = "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the contents of the image in detail:"
    
    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,  # Recommended for Qwen3-VL Instruct
            top_p=0.8,
            top_k=20,
        )
    
    # Decode
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


# ==================== Main Execution ====================

def main():
    """
    Example usage with configurable parameters for Qwen3-VL-8B-Instruct
    """
    
    # Configure model
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype="bfloat16",
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Configure LoRA - Language attention only (recommended starting point)
    lora_config = LoRAConfig(
        r=8,  # Research suggests rank 8-16 works well for Qwen3 [web:5]
        lora_alpha=16,
        lora_dropout=0.05,
        target_preset="language_attention",
        include_vision_encoder=False,
        include_merger=False,
        include_language_model=True,
    )
    
    # Alternative configurations:
    
    # Full attention (vision + language)
    # lora_config = LoRAConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_preset="attention_all",
    #     include_vision_encoder=True,
    #     include_language_model=True,
    # )
    
    # Full model (attention + MLP + merger)
    # lora_config = LoRAConfig(
    #     r=32,
    #     lora_alpha=64,
    #     target_preset="full_model",
    #     include_vision_encoder=True,
    #     include_merger=True,
    #     include_language_model=True,
    # )
    
    # Custom modules
    # lora_config = LoRAConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_preset="custom",
    #     custom_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # )
    
    # Configure dataset
    dataset_config = DatasetConfig(
        dataset_name="VRSBench",
        task="caption",  # "vqa", "caption", or "both"
        max_length=2048,
    )
    
    # Configure training
    training_config = TrainingConfig(
        output_dir="./qwen3_vl_vrsbench_lora",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
    )
    
    # Train
    model, processor, trainer = train_vlm(
        model_config,
        lora_config,
        dataset_config,
        training_config,
    )


if __name__ == "__main__":
    main()
