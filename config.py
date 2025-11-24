"""
Configuration for VLM adapter training with distributed support.
Supports any frozen vision encoder and language decoder.
"""
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class ModelConfig:
    """Configuration for vision encoder, language decoder, and adapter."""
    # Vision encoder settings
    vision_encoder_name: str = "moonshotai/Kimi-VL-A3B-Instruct"  # Any HF vision model or VLM's vision tower
    freeze_vision_encoder: bool = True
    
    # Language decoder settings
    language_decoder_name: str = "Qwen/Qwen3-VL-8B-Instruct"  # Any HF VLM'S LM or LM
    freeze_language_decoder: bool = True
    
    # Model dimensions (auto-detected from pretrained models)
    vision_hidden_size: Optional[int] = None
    language_hidden_size: Optional[int] = None
    

@dataclass
class DataConfig:
    """Configuration for VQA dataset loading."""
    dataset_names: List[str] = field(default_factory=lambda: ["HuggingFaceM4/VQAv2"])
    dataset_splits: List[str] = field(default_factory=lambda: ["train"])
    val_dataset_name: str = "HuggingFaceM4/VQAv2"
    val_split: str = "validation"
    
    # Data processing
    max_length: int = 512
    image_size: int = 224
    num_workers: int = 4
    
    # Streaming and caching
    streaming: bool = False
    cache_dir: Optional[str] = None
    

@dataclass
class TrainConfig:
    """Training configuration with distributed support."""
    # Training hyperparameters
    batch_size: int = 16  # Per-GPU batch size
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    distributed_backend: str = "nccl"  # nccl for GPU, gloo for CPU
    find_unused_parameters: bool = False
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Logging
    wandb_project: Optional[str] = "vlm-adapter-training"
    wandb_run_name: Optional[str] = None
    log_to_wandb: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass
class VLMAdapterTrainConfig:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
