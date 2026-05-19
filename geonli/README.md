# GeoNLI

A **pip-installable**, **pluggable** toolkit for remote-sensing image analysis:
- **Image Captioning**
- **Object Grounding** (oriented bounding boxes)
- **Visual Question Answering** (binary, numeric, semantic)

## Philosophy

- **Any model**: Swap VLMs (Qwen3-VL, LLaVA, GPT-4V, Gemini, …) and segmenters (SAM3, SAM2, Grounding-DINO, …) via a unified registry.
- **Any dataset**: JSON, folder-based, or custom loaders — all through one config file.
- **Any setup**: Local GPU, cloud API-only, or hybrid — no code changes required.

## Quick Start

```bash
# 1. Install
pip install geonli

# 2. (Optional) Install backend extras
pip install geonli[transformers,sam]

# 3. Write a config (see configs/default.yaml)
# 4. Run
geonli-run --config my_config.yaml
```

## Programmatic Usage

```python
from geonli import get_vlm, get_segmenter, get_task
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline

# 1. Pick components by registered name
vlm = get_vlm("dummy")
seg = get_segmenter("dummy")

# 2. Build tasks
caption = get_task("captioning", vlm=vlm)
ground = get_task("grounding", vlm=vlm, segmenter=seg)
vqa = get_task("vqa", vlm=vlm, segmenter=seg)

# 3. Assemble pipeline
pipeline = DefaultGeoNLIPipeline(tasks=[caption, ground, vqa])

# 4. Run on a PIL image
from PIL import Image
img = Image.open("satellite.png")
results = pipeline.run(
    image=img,
    queries={
        "caption_query": {"instruction": "Describe the image."},
        "grounding_query": {"instruction": "Locate all buildings."},
        "attribute_query": {
            "binary": {"instruction": "Is there water?"},
            "numeric": {"instruction": "How many cars?"},
        }
    },
    metadata={"spatial_resolution_m": 1.57},
)
print(results["captioning"].response)
```

## Config-Driven Execution

```yaml
# my_config.yaml
models:
  vlm:
    name: "isro-qwen3-vl"
    model_id: "Dinosaur2314/qwen_finetune11"
  segmenter:
    name: "isro-sam3"
    model_id: "facebook/sam3"

tasks:
  - name: "captioning"
  - name: "grounding"
  - name: "vqa"

dataset:
  name: "json_dataset"
  input_json: "query.json"
```

Then run:
```bash
geonli-run --config my_config.yaml
```

## Architecture

| Layer | Responsibility |
|-------|---------------|
| `core` | Abstract base classes (`VLMBase`, `SegmenterBase`, `TaskBase`, `GeoNLIPipeline`) + Registry + Config |
| `models` | Concrete VLM / segmenter backends (transformers, API, dummy) |
| `tasks` | Generic `CaptioningTask`, `GroundingTask`, `VQATask` |
| `datasets` | JSON / folder loaders conforming to `GeoNLIDataset` |
| `prompts` | YAML/JSON prompt banks loaded by `PromptManager` |
| `adapters` | Bridges to existing monolithic code (e.g., `isro_geonli.py`) |
| `cli` | `geonli-run`, `geonli-eval`, etc. |

## Adding a Custom Model (3 steps)

```python
from geonli import VLMBase, register_vlm

@register_vlm("my-model")
class MyModel(VLMBase):
    def __init__(self, model_id: str, **kwargs):
        ...
    def query(self, image, prompt, **kwargs) -> str:
        ...
    def model_name(self) -> str:
        return "my-model"
```

That’s it — the model is now available in any config via `name: "my-model"`.

## License

MIT
