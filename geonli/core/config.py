"""
Pydantic-based configuration schemas.
Enables validation, auto-complete, and easy serialization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    model_id: Optional[str] = None
    device: str = "cuda"
    torch_dtype: Optional[str] = "float16"
    api_key_env: Optional[str] = None   # e.g. "OPENAI_API_KEY" for API-based VLMs
    extra: Dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    name: str
    enabled: bool = True
    prompt_template: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    fallback_to_vlm: bool = True
    score_threshold: float = 0.4
    extra: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    input_json: Optional[str] = None
    image_root: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    save_dir: str = "./outputs"
    format: str = "json"          # json, jsonl, csv
    log_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class ExperimentConfig(BaseModel):
    experiment_name: str = "geonli_experiment"
    device: str = "cuda"
    seed: int = 42
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    tasks: List[TaskConfig] = Field(default_factory=list)
    dataset: Optional[DatasetConfig] = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        import json
        with open(path, "r") as f:
            raw = json.load(f)
        return cls(**raw)
