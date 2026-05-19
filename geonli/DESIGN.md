# GeoNLI Package Design Document

## Goal
Transform the monolithic ISRO-GeoNLI pipeline into a pip-installable, pluggable package that supports:
- **Any VLM** (Qwen3-VL, LLaVA, InternVL, GPT-4V, Gemini, etc.)
- **Any segmentation / grounding model** (SAM3, SAM2, SAM, Grounding-DINO, etc.)
- **Any dataset** (via registry & configurable loaders)
- **Any setup** (local GPU, API-only, mixed, edge devices)

## Guiding Principles
1. **Inversion of Control**: The pipeline asks the user for a `Task` + `Model` combination; it does not hardcode them.
2. **Registry Pattern**: Models, tasks, datasets, and prompts are registered by name and instantiated from config.
3. **Config-Driven**: A single YAML/JSON config file defines the entire experiment (model IDs, task list, dataset paths, prompts).
4. **Backend Agnostic**: Support both `transformers`-based local models and REST API-based cloud models through a unified interface.
5. **Backward Compatibility**: Existing ISRO-GeoNLI code lives in `adapters/` and can be used out-of-the-box.

## Package Layout

```
geonli/
├── core/
│   ├── base.py          # Abstract base classes (Task, Model, Pipeline, Dataset)
│   ├── registry.py      # Global registries for plugins
│   ├── config.py        # Pydantic config schemas
│   └── pipeline.py      # Generic GeoNLIPipeline orchestrator
├── models/
│   ├── base.py          # VLMBase, SegmenterBase
│   ├── transformers_models.py  # Qwen3-VL, LLaVA, etc.
│   ├── api_models.py    # OpenAI, Gemini, Claude adapters
│   └── sam_backends.py  # SAM3, SAM2, dummy segmenter
├── tasks/
│   ├── base.py          # Task base classes
│   ├── captioning.py    # Generic CaptioningTask
│   ├── grounding.py     # Generic GroundingTask
│   └── vqa.py           # Generic VQATask with pluggable router & agent
├── datasets/
│   ├── base.py          # Dataset base + collators
│   ├── json_dataset.py  # Load from JSON/JSONL (GeoNLI format)
│   └── image_folder.py  # Folder-based dataset
├── prompts/
│   ├── manager.py       # Prompt template manager
│   └── templates/       # YAML prompt banks
├── configs/
│   ├── default.yaml
│   └── examples/
├── cli/
│   ├── run.py           # geonli-run
│   ├── train.py         # geonli-train
│   └── eval.py          # geonli-eval
├── adapters/            # Bridges to existing ISRO-GeoNLI code
│   └── isro_geonli.py
└── utils/
    └── viz.py
```

## Core Abstractions

### 1. Model Interface
```python
class VLMBase(ABC):
    @abstractmethod
    def query(self, image: Image.Image | None, prompt: str, **kwargs) -> str: ...

class SegmenterBase(ABC):
    @abstractmethod
    def segment(self, image: Image.Image, text_prompt: str, **kwargs) -> SegmentationResult: ...
```

### 2. Task Interface
```python
class TaskBase(ABC):
    @abstractmethod
    def run(self, image: Image.Image, query: str, context: dict | None = None) -> TaskResult: ...
```

### 3. Dataset Interface
```python
class GeoNLIDataset(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, idx) -> dict:
        # Must return {"image": PIL.Image, "queries": {...}, "metadata": {...}}
```

### 4. Registry
All components register themselves:
```python
@register_vlm("qwen3-vl-8b")
class Qwen3VL(VLMBase): ...

@register_task("captioning")
class CaptioningTask(TaskBase): ...
```

## Configuration Schema (Pydantic)

```yaml
# config.yaml
experiment_name: "my_geonli_run"

device: "cuda"
seed: 42

models:
  vlm:
    name: "qwen3-vl-8b"          # registered name
    model_id: "Qwen/Qwen3-VL-8B-Instruct"
    device: "cuda"
    torch_dtype: "float16"
  segmenter:
    name: "sam3"
    model_id: "facebook/sam3"
    device: "cuda"

tasks:
  - name: "captioning"
    enabled: true
    prompt_template: "default_caption"
    max_tokens: 512
  - name: "grounding"
    enabled: true
    score_threshold: 0.4
    fallback_to_vlm: true
  - name: "vqa"
    enabled: true
    router_type: "auto"          # auto, sam, vlm

dataset:
  name: "json_dataset"
  input_json: "path/to/query.json"
  image_root: "path/to/images/"
  batch_size: 1

output:
  save_dir: "./outputs"
  format: "json"                # json, csv, wandb
```

## CLI Usage

```bash
# Install
pip install geonli

# Run inference from config
geonli-run --config configs/my_exp.yaml

# Run with overrides
geonli-run --config configs/default.yaml \
           --override models.vlm.name=gemini-1.5-flash \
           --override tasks.grounding.enabled=false

# Train / Fine-tune
geonli-train --config configs/train_lora.yaml --dataset my_dataset

# Evaluate
geonli-eval --preds outputs/preds.json --gt outputs/gt.json
```

## Migration Strategy for Existing Code

1. **Phase 1**: Build `geonli/core/` abstract layer and `geonli/adapters/isro_geonli.py` that wraps the current `rs_pipeline.py`, ` tasks/`, and `model/` code.
2. **Phase 2**: Refactor hardcoded prompts into `geonli/prompts/templates/`.
3. **Phase 3**: Add new model backends (API-only VLMs, SAM2) without touching task logic.
4. **Phase 4**: Move dataset curation scripts into `geonli/datasets/` as registered loaders.

## Why This Design?

| Problem in Current Code | Package Solution |
|------------------------|------------------|
| Hardcoded Qwen3-VL + SAM3 | `VLMBase` / `SegmenterBase` with registry |
| Prompts scattered in `.py` files | `PromptManager` loading YAML templates |
| No dataset abstraction | `GeoNLIDataset` base + JSON / folder / HuggingFace loaders |
| Monolithic pipeline | `GeoNLIPipeline` injects tasks; tasks inject models |
| Hard to add a new model | Implement 2 methods (`query`, `segment`) + `@register_*` |
| No CLI / config-driven runs | Hydra / Pydantic configs + `geonli-run` entry point |
