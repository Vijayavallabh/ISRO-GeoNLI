# GeoNLI Design Document

## Goal

A pip-installable, pluggable Python package for remote-sensing image analysis that supports **any HuggingFace VLM**, **any segmenter**, **any dataset**, and **any setup** — all driven by a single YAML config file.

**Status:** Implemented and tested on A100 with `Qwen/Qwen3-VL-8B-Instruct` + `facebook/sam3`.

---

## Guiding Principles

1. **Inversion of Control** — The user assembles `Task` + `Model` combinations; the pipeline does not hardcode them.
2. **Registry Pattern** — Models, tasks, datasets, and prompts are registered by string name and instantiated from config.
3. **Config-Driven** — One YAML/JSON file defines the entire experiment.
4. **Backend Agnostic** — Works with local `transformers` models and REST API-based cloud models through a unified `VLMBase` interface.
5. **Prompt-First** — Every system prompt, router instruction, and agent example lives in external `.txt` files. Zero hardcoded strings in Python.
6. **Backward Compatible** — Existing ISRO-GeoNLI code lives in `adapters/` and is usable out-of-the-box.

---

## Package Layout

```
geonli/
├── core/
│   ├── base.py              # Abstract contracts: VLMBase, TransformersVLMBase,
│   │                        #   SegmenterBase, TaskBase, AgentBase, GeoNLIPipeline
│   ├── registry.py          # @register_* decorators + get_* lookup functions
│   ├── config.py            # Pydantic ExperimentConfig with YAML/JSON loaders
│   └── pipeline_impl.py     # DefaultGeoNLIPipeline (wires tasks + shared context)
├── models/
│   ├── base.py              # DummyVLM, DummySegmenter (for CI/testing)
│   ├── huggingface.py       # HuggingFaceVLM: auto-detects Qwen/LLaVA/InternVL/Idefics
│   ├── huggingface_sam.py   # HuggingFaceSAM: SAM-vit-huge / SAM2 via AutoProcessor
│   └── huggingface_sam3.py  # HuggingFaceSAM3: facebook/sam3 (text-prompted)
├── tasks/
│   ├── captioning.py        # CaptioningTask (external prompt template)
│   ├── grounding.py         # GroundingTask (extraction → segment → OBB + geometry →
│   │                        #   VLM selection → fallback → visualization)
│   ├── vqa.py               # VQATask (LLM router + VLM path / SAM+Agent path)
│   └── agent.py             # TransformersSatelliteAgent (4 tools, multi-step loop)
├── datasets/
│   ├── base.py              # JsonGeoNLIDataset (GeoNLI JSON format)
│   ├── huggingface_dataset.py  # Any HF datasets.Dataset
│   ├── csv_dataset.py       # CSV/TSV with configurable columns
│   └── image_folder.py      # Images + JSON sidecars
├── prompts/
│   ├── manager.py           # PromptManager loads .txt / .yaml / .json banks
│   └── templates/           # All external prompt files (zero hardcoded fallbacks)
│       ├── default_caption.txt
│       ├── default_grounding_extraction.txt
│       ├── default_router.txt
│       ├── vqa_numeric.txt
│       ├── vqa_binary.txt
│       ├── vqa_semantic.txt
│       └── satellite_agent.txt
├── configs/
│   ├── default.yaml         # Dummy-mode config (no GPU needed)
│   ├── example_hf.yaml      # Real HF VLM (Qwen3-VL) config
│   └── example_isro.yaml    # Backward-compatible ISRO adapter config
├── adapters/
│   └── isro_geonli.py       # Bridges to existing ISRO-GeoNLI monolithic code
├── cli/
│   └── run.py               # geonli-run --config <path> [--prompt-dir <path>]
└── utils/
    ├── geo.py               # mask_to_obb (OpenCV-based, no model deps)
    └── satellite_tools.py   # select_object_by_rank, calculate_distance_by_indices,
                             #   calculator_tool (used by agent)
```

---

## Core Abstractions

### 1. VLMBase — Any Vision-Language Model

```python
class VLMBase(ABC):
    @abstractmethod
    def query(self, image: Image.Image | None, prompt: str,
              system_prompt: str | None = None, max_tokens: int = 512,
              temperature: float = 0.0, **kwargs) -> str: ...

class TransformersVLMBase(VLMBase):
    model: Any          # raw HF model
    processor: Any      # raw HF processor
    device: Any
    @abstractmethod
    def chat_generate(self, messages: list[dict], max_new_tokens: int = 512,
                      temperature: float = 0.0, **kwargs) -> str: ...
```

`TransformersVLMBase` exposes raw `.model` / `.processor` needed by the multi-turn tool-calling agent.

### 2. SegmenterBase — Any Segmentation Model

```python
class SegmenterBase(ABC):
    @abstractmethod
    def segment(self, image: Image.Image, text_prompt: str,
                **kwargs) -> SegmentationResult | None: ...
```

### 3. TaskBase — Any Task

```python
class TaskBase(ABC):
    name: str
    @abstractmethod
    def run(self, image: Image.Image, query: str,
            context: dict | None = None) -> TaskResult: ...
```

### 4. Registry — Plugin-Style Extension

```python
@register_vlm("huggingface")
class HuggingFaceVLM(TransformersVLMBase): ...

@register_task("grounding")
class GroundingTask(TaskBase): ...

@register_dataset("csv_dataset")
class CSVGeoNLIDataset(GeoNLIDataset): ...
```

Components are then instantiated from config by name:
```python
vlm = get_vlm("huggingface", model_id="Qwen/Qwen3-VL-8B-Instruct")
```

---

## GroundingTask Pipeline (Tested)

1. **Target Extraction** — VLM turns user description into a noun phrase via external prompt template.
2. **Segmentation** — Segmenter receives `"tree"` and returns N binary masks.
3. **OBB + Geometric Features** — For each mask:
   - `cv2.findContours` → `cv2.minAreaRect` → 8-point OBB
   - Width, Height, Area, Aspect Ratio, Angle, Compactness
   - Normalized relative coordinates (0–1000)
4. **Annotated Image** — Colored mask overlays + numeric IDs + 10×10 magenta/cyan grid
5. **VLM Selection** — Annotated image + geometric features fed back to VLM; VLM returns mask IDs
6. **Fallback** — If segmenter returns 0 masks, VLM Direct Localization parses coordinates directly
7. **Visualization** (optional) — Matplotlib with green final boxes + yellow dashed candidates

**Test result on A100** (SAM3 + Qwen3-VL, query `"Locate all trees."`):
- SAM3: 19 masks detected
- OBB extraction: all valid
- VLM selection: 4 best trees chosen
- Output: 4 `Detection` objects with real OBB coordinates

---

## VQATask Pipeline (Tested)

### Routing

| Router | Requires | Used When |
|--------|----------|-----------|
| `LLMRouter` | Any `VLMBase` (via `.query()`) | External prompt template `default_router.txt` is registered |
| `KeywordRouter` | None | Fallback if template missing |

Router decides `"SAM"` or `"VLM"` per question.

### VLM Path

Direct `.query()` with type-specific external system prompt (`vqa_numeric.txt`, `vqa_binary.txt`, `vqa_semantic.txt`).

### SAM + Agent Path

`TransformersSatelliteAgent` runs a multi-step loop:

- **Tool schema** (external `satellite_agent.txt`):
  - `detect_objects(target_class)`
  - `get_object_info(sort_attribute, rank_index, return_attribute)`
  - `measure_distance(index_1, index_2)`
  - `calculate(expression)`

- **Loop**: VLM generates → parse JSON tool call → execute → append observation → repeat until final answer.

- **Zero-object handling**: Agent correctly answers `0` / `No` when `detect_objects` returns empty.

**Test result on A100** (Qwen3-VL):
- `"How many buildings?"` → Agent → `detect_objects("building")` → `0`
- `"Is there water?"` → Agent → `detect_objects("water body")` → `No`

---

## HuggingFaceVLM: Family Auto-Detection

```python
vlm = get_vlm("huggingface", model_id="Qwen/Qwen3-VL-8B-Instruct")
# Detects "qwen" → uses qwen_vl_utils.process_vision_info()

vlm = get_vlm("huggingface", model_id="liuhaotian/llava-v1.6-vicuna-7b")
# Detects "llava" → uses LLaVA-style processor(text, images=...)
```

Supported families: `qwen`, `llava`, `internvl`, `idefics`, `generic`.

Architecture override:
```yaml
models:
  vlm:
    name: "huggingface"
    model_id: "..."
    architecture: "llava"   # force detection
```

---

## Prompt-First Design

**Rule:** If a prompt template is referenced in config but missing from the prompt bank, the code **raises `KeyError`** — never silently falls back to a hardcoded string.

Implementation:
- `PromptManager` scans `geonli/prompts/templates/*.txt` at startup
- `get_prompt("default_caption")` returns the raw string
- Tasks call `get_prompt(self.prompt_template)` directly

Custom prompts:
```bash
geonli-run --config my_config.yaml --prompt-dir ./my_custom_prompts/
```
Custom `.txt` files in `./my_custom_prompts/` override or extend built-ins.

---

## Config Schema (Pydantic)

```python
class ExperimentConfig(BaseModel):
    experiment_name: str
    device: str
    seed: int
    models: dict[str, ModelConfig]
    tasks: list[TaskConfig]
    dataset: DatasetConfig | None
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig": ...
```

Loaded from YAML with optional `--override key.subkey=value` CLI flags.

---

## Backward Compatibility

The `adapters/isro_geonli.py` module wraps the existing monolithic code:

- `ISRO_VLM` → wraps `VLMInterface` (Qwen3-VL + LoRA)
- `ISRO_Segmenter` → wraps `SAM3Interface`
- `ISROCaptioningTask`, `ISROGroundingTask`, `ISROVQATask` → thin wrappers around original task classes

This means existing users can migrate to the new config-driven package gradually without rewriting their models.

---

## Why This Design?

| Problem in Monolithic ISRO-GeoNLI | Package Solution |
|-----------------------------------|------------------|
| Hardcoded Qwen3-VL + SAM3 | `VLMBase` / `SegmenterBase` + registry |
| Prompts scattered in `.py` files | `PromptManager` loading external `.txt` templates |
| No dataset abstraction | `GeoNLIDataset` base + JSON / HF / CSV / folder loaders |
| Monolithic pipeline | `GeoNLIPipeline` injects tasks; tasks inject models |
| Adding a new model requires editing tasks | Implement 2 methods + `@register_*` |
| No CLI / config-driven runs | `geonli-run --config <yaml>` with Pydantic validation |
