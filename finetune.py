"""
Modular LoRA Fine-tuning Script for Vision-Language Models
Supports flexible model selection, component-specific LoRA application, and dataset switching
Optimized for Kimi-VL architecture (MoonViT + DeepseekV3)
"""
from dataloader import build_vrs_dataloaders_train_test_only
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import transformers.activations as activations
from transformers.activations import GELUTanh
activations.PytorchGELUTanh = GELUTanh
PytorchGELUTanh = GELUTanh

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
    model_name: str = "moonshotai/Kimi-VL-A3B-Instruct"
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
    """Configuration for LoRA fine-tuning with Kimi-VL specific architecture support"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM
    
    # Target modules configuration
    target_modules: Optional[List[str]] = None
    
    # Preset options for Kimi-VL architecture
    target_preset: Literal[
        "language_attention",           # Only LLM attention layers
        "language_mlp",                 # Only LLM MLP/MoE layers
        "language_all",                 # All LLM components
        "vision_attention",             # Only vision encoder attention
        "vision_mlp",                   # Only vision encoder MLP
        "vision_all",                   # All vision encoder components
        "projector",                    # Multi-modal projector only
        "attention_all",                # All attention layers (vision + language)
        "mlp_all",                      # All MLP layers (vision + language)
        "full_model",                   # Everything except embeddings/lm_head
        "custom"                        # Use custom_modules
    ] = "language_attention"
    
    # Component selection for fine-grained control
    include_vision_encoder: bool = False
    include_projector: bool = False
    include_language_model: bool = True
    
    # MoE-specific options (for DeepseekV3 layers 1-26)
    apply_lora_to_moe_experts: bool = False  # Apply LoRA to MoE expert MLPs
    apply_lora_to_moe_shared: bool = True    # Apply LoRA to shared experts
    apply_lora_to_moe_gate: bool = False     # Apply LoRA to MoE gating
    
    # Custom module names (used when target_preset="custom")
    custom_modules: List[str] = field(default_factory=list)
    
    # Module pattern exclusions (optional)
    modules_to_exclude: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    
    def get_target_modules(self, model) -> List[str]:
        """
        Determine target modules based on Kimi-VL architecture
        
        Model structure:
        - vision_tower.encoder.blocks[*].{wqkv, wo, mlp.fc0, mlp.fc1}
        - multi_modal_projector.{linear_1, linear_2}
        - language_model.model.layers[*].self_attn.{q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj}
        - language_model.model.layers[0].mlp.{gate_proj, up_proj, down_proj}  # Dense layer
        - language_model.model.layers[1-26].mlp.experts[*].{gate_proj, up_proj, down_proj}  # MoE
        - language_model.model.layers[1-26].mlp.shared_experts.{gate_proj, up_proj, down_proj}
        """
        if self.target_modules is not None:
            return self.target_modules
            
        if self.target_preset == "custom":
            return self.custom_modules
        
        # Define module groups for Kimi-VL
        vision_attention = ["wqkv", "wo"]  # Vision encoder uses combined qkv
        vision_mlp = ["fc0", "fc1"]
        
        projector_modules = ["linear_1", "linear_2"]
        
        # DeepseekV3 uses Multi-head Latent Attention (MLA)
        language_attention = [
            "q_proj",                # Query projection
            "kv_a_proj_with_mqa",   # Key-Value latent compression
            "kv_b_proj",            # Key-Value expansion
            "o_proj"                # Output projection
        ]
        
        language_mlp = ["gate_proj", "up_proj", "down_proj"]
        
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
            
        elif self.target_preset == "projector":
            targets = projector_modules
            
        elif self.target_preset == "attention_all":
            targets = vision_attention + language_attention
            
        elif self.target_preset == "mlp_all":
            targets = vision_mlp + language_mlp
            
        elif self.target_preset == "full_model":
            targets = (vision_attention + vision_mlp + 
                      projector_modules + 
                      language_attention + language_mlp)
        
        # Apply component filters
        if not self.include_vision_encoder:
            targets = [t for t in targets if t not in vision_attention + vision_mlp]
        
        if not self.include_projector:
            targets = [t for t in targets if t not in projector_modules]
            
        if not self.include_language_model:
            targets = [t for t in targets if t not in language_attention + language_mlp]
        
        return targets
    
    def get_modules_to_save(self) -> Optional[List[str]]:
        """
        Specify additional modules to save during training
        Useful for projector or embedding layers
        """
        modules = []
        
        if self.include_projector:
            modules.extend(["multi_modal_projector"])
            
        return modules if modules else None
    
    def to_peft_config(self, model) -> LoraConfig:
        """Convert to PEFT LoraConfig object"""
        target_modules = self.get_target_modules(model)
        modules_to_save = self.get_modules_to_save()
        
        print(f"\n{'='*60}")
        print(f"LoRA Configuration:")
        print(f"  Rank: {self.r}, Alpha: {self.lora_alpha}, Dropout: {self.lora_dropout}")
        print(f"  Target preset: {self.target_preset}")
        print(f"  Target modules: {target_modules}")
        print(f"  Vision Encoder: {self.include_vision_encoder}")
        print(f"  Projector: {self.include_projector}")
        print(f"  Language Model: {self.include_language_model}")
        if self.apply_lora_to_moe_experts or self.apply_lora_to_moe_shared:
            print(f"  MoE Experts: {self.apply_lora_to_moe_experts}")
            print(f"  MoE Shared: {self.apply_lora_to_moe_shared}")
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
        print("Model Module Analysis")
        print("="*60 + "\n")
        
        vision_modules = set()
        projector_modules = set()
        language_modules = set()
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]
                
                if 'vision_tower' in name:
                    vision_modules.add(module_name)
                elif 'multi_modal_projector' in name:
                    projector_modules.add(module_name)
                elif 'language_model' in name:
                    language_modules.add(module_name)
        
        print("Vision Encoder Linear Modules:")
        print(f"  {sorted(vision_modules)}\n")
        
        print("Projector Linear Modules:")
        print(f"  {sorted(projector_modules)}\n")
        
        print("Language Model Linear Modules:")
        print(f"  {sorted(language_modules)}\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        vision_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'vision_tower' in n)
        projector_params = sum(p.numel() for n, p in model.named_parameters() 
                              if 'multi_modal_projector' in n)
        language_params = sum(p.numel() for n, p in model.named_parameters() 
                             if 'language_model' in n)
        
        print("Parameter Distribution:")
        print(f"  Total: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Vision: {vision_params:,} ({vision_params/total_params*100:.1f}%)")
        print(f"  Projector: {projector_params:,} ({projector_params/total_params*100:.1f}%)")
        print(f"  Language: {language_params:,} ({language_params/total_params*100:.1f}%)")
        print("="*60 + "\n")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing"""
    dataset_name: str = "VRSBench"
    task: Literal["vqa", "caption", "both"] = "both"
    streaming: bool = False
    max_samples: Optional[int] = None
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
    image_size: Optional[int] = None  # None means use processor default


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
    Custom data collator for VLMs that handles flattening of pixel_values
    and other image-related tensors that shouldn't be stacked normally.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        first = features[0]
        
        # Keys that should be concatenated (flattened batch) instead of stacked
        # pixel_values: [N, C, H, W] -> [B*N, C, H, W]
        # grid_hws: [N, 2] -> [B*N, 2] (Kimi-VL specific)
        concat_keys = ["pixel_values", "grid_hws", "image_grid_thw", "pixel_attention_mask"]
        
        for k, v in first.items():
            if k in concat_keys:
                # Concatenate along batch dimension (flattening the batch of lists/tensors)
                batch[k] = torch.cat([f[k] for f in features], dim=0)
            else:
                # Default stacking for text and other tensors
                try:
                    batch[k] = torch.stack([f[k] for f in features])
                except Exception:
                    # Fallback for non-tensor data
                    batch[k] = [f[k] for f in features]
                    
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
        
        # Apply max_samples limit if specified
        if self.config.max_samples and len(self.flattened) > self.config.max_samples:
            self.flattened = self.flattened[:self.config.max_samples]
    
    def __len__(self) -> int:
        return len(self.flattened)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flattened[idx]
        image = item["image"]
        task = item["task"]
        image_token = "<|media_pad|>"
        
        if task == "vqa":
            question = item["question"]
            answer = item["answer"]
            prompt = f"{image_token}\nQuestion: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"
        elif task == "caption":
            caption = item["text"]
            prompt = f"{image_token}\nDescribe the contents of the image in detail:"
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
        
        # Remove batch dimension added by processor
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Create labels (same as input_ids for causal LM, with prompt masked)
        labels = encoding["input_ids"].clone()
        
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
        if "attention_mask" in encoding:
            labels[encoding["attention_mask"] == 0] = -100
        
        encoding["labels"] = labels
        
        return encoding


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
        
        # Apply max_samples limit if specified
        if self.config.max_samples and len(self.dataset) > self.config.max_samples:
            self.dataset = self.dataset.select(range(self.config.max_samples))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Process a single example for VQA or captioning
        """
        item = self.dataset[idx]
        
        # Load image
        image = item[self.config.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        image_token = "<|media_pad|>"

        if self.config.task == "vqa":
            question = item[self.config.vqa_question_column]
            answer = item[self.config.vqa_answer_column]
            prompt = f"{image_token}\nQuestion: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"
        
        elif self.config.task == "caption":
            caption = item[self.config.caption_column]
            prompt = f"{image_token}\nDescribe the contents of the image in detail:"
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
        
        # Remove batch dimension added by processor
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Create labels (same as input_ids for causal LM, with prompt masked)
        labels = encoding["input_ids"].clone()
        
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
        if "attention_mask" in encoding:
            labels[encoding["attention_mask"] == 0] = -100
        
        encoding["labels"] = labels
        
        return encoding


# ==================== Model Setup ====================

def load_model_and_processor(model_config: ModelConfig):
    """
    Load VLM model and processor with optional quantization
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_config.model_name}")
    print(f"{'='*60}\n")
    
    # Load processor/tokenizer
    try:
        processor = AutoProcessor.from_pretrained(
            model_config.model_name,
            trust_remote_code=model_config.trust_remote_code
        )
    except:
        # Fallback to tokenizer if processor not available
        processor = AutoTokenizer.from_pretrained(
            model_config.model_name,
            trust_remote_code=model_config.trust_remote_code
        )
    
    # Setup quantization
    quantization_config = model_config.get_quantization_config()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
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
        
        # Manually enable gradient checkpointing on submodules that support it
        print("Enabling gradient checkpointing on supported submodules...")
        
        # Enable for language model (DeepseekV3)
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'gradient_checkpointing_enable'):
            try:
                model.language_model.gradient_checkpointing_enable()
                print("  ✓ Enabled gradient checkpointing for language_model")
            except Exception as e:
                print(f"  ✗ Could not enable for language_model: {e}")
        
        # Enable for vision tower if it supports it
        if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'gradient_checkpointing_enable'):
            try:
                model.vision_tower.gradient_checkpointing_enable()
                print("  ✓ Enabled gradient checkpointing for vision_tower")
            except Exception as e:
                print(f"  ✗ Could not enable for vision_tower: {e}")
    
    # Enable gradient checkpointing for inputs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Manual implementation if method doesn't exist
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        print("  ✓ Manually enabled input gradients")

       
    # Ensure model and tokenizer are in sync with special tokens
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    # Updated tokens: {'eos_token_id': 163585, 'bos_token_id': 163584, 'pad_token_id': 163838}
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 163838
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = 163584
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = 163585
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer Special Tokens: PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")
    print(f"\nModel loaded successfully!\n")
    
    return model, processor


def apply_lora(model, lora_config: LoRAConfig):
    """
    Apply LoRA adapters to the model with Kimi-VL specific handling
    """
    print("\n" + "="*60)
    print("Applying LoRA to Kimi-VL Model")
    print("="*60 + "\n")
    
    # Optional: Inspect model structure first
    # Uncomment to see detailed module breakdown
    #lora_config.inspect_model_modules(model)
    
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
                          if p.requires_grad and 'vision_tower' in n)
    projector_trainable = sum(p.numel() for n, p in model.named_parameters() 
                             if p.requires_grad and 'multi_modal_projector' in n)
    language_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if p.requires_grad and 'language_model' in n)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if total_trainable > 0:
        print("\nComponent Breakdown:")
        if vision_trainable > 0:
            print(f"  Vision Encoder: {vision_trainable:,} "
                  f"({vision_trainable/total_trainable*100:.2f}%)")
        if projector_trainable > 0:
            print(f"  Projector: {projector_trainable:,} "
                  f"({projector_trainable/total_trainable*100:.2f}%)")
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
        prompt = f"Question: {question}\nAnswer:"
    else:
        prompt = "Describe the contents of the image in detail:"
    
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
            temperature=0.2,  # Recommended for Instruct model
            top_p=0.9,
        )
    
    # Decode
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


# ==================== Main Execution ====================

def main():
    """
    Example usage with configurable parameters
    """
    
    # Configure model
    model_config = ModelConfig(
        model_name="moonshotai/Kimi-VL-A3B-Instruct",
        torch_dtype="bfloat16",
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Configure LoRA - Language attention only (recommended starting point)
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_preset="language_attention",
        include_vision_encoder=False,
        include_projector=False,
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
    
    # Full model (attention + MLP + projector)
    # lora_config = LoRAConfig(
    #     r=32,
    #     lora_alpha=64,
    #     target_preset="full_model",
    #     include_vision_encoder=True,
    #     include_projector=True,
    #     include_language_model=True,
    # )
    
    # Custom modules
    # lora_config = LoRAConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_preset="custom",
    #     custom_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
    # )
    
    # Configure dataset
    dataset_config = DatasetConfig(
        dataset_name="VRSBench",
        task="both",  # "vqa", "caption", or "both"
        max_samples=None,  # Set to int for quick testing
        max_length=2048,
    )
    
    # Configure training
    training_config = TrainingConfig(
        output_dir="./kimi_vl_vrsbench_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
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
    
    # Example inference
    # result = inference_example(
    #     model, processor, 
    #     "test_image.jpg", 
    #     "What objects are visible in this image?"
    # )
    # print(result)


if __name__ == "__main__":
    main()
