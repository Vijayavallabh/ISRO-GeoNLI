# GeoNLI

A **pip-installable**, **pluggable** toolkit for remote-sensing image analysis:
- **Image Captioning**
- **Object Grounding** (oriented bounding boxes)
- **Visual Question Answering** (binary, numeric, semantic)

## Philosophy

- **Any HuggingFace VLM**: Qwen3-VL, LLaVA, InternVL, Idefics, or generic `AutoModelForVision2Seq` — just provide a `model_id`.
- **Any segmenter**: SAM3, SAM2, Grounding-DINO, or your own — implement two methods and register.
- **Any dataset**: GeoNLI JSON, HuggingFace `datasets`, CSV/TSV, or image-folder + JSON sidecars — all through one config.
- **Any setup**: Local GPU, cloud API, edge device — no code changes required.
- **Every prompt is external**: All system prompts, router prompts, and agent instructions live in text files. Users change behavior by editing YAML/txt, not Python.

## Quick Start

```bash
# 1. Install
pip install geonli

# 2. (Optional) Install backend extras
pip install geonli[transformers,sam]

# 3. Run inference from config
geonli-run --config geonli/configs/example_hf.yaml

# 4. Use custom prompts
geonli-run --config my_config.yaml --prompt-dir ./my_prompts/
```

## Programmatic Usage

```python
from geonli import get_vlm, get_segmenter, get_task
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline

# 1. Pick components by registered name
vlm     = get_vlm("huggingface", model_id="Qwen/Qwen3-VL-8B-Instruct")
seg     = get_segmenter("dummy")

# 2. Build tasks
caption = get_task("captioning", vlm=vlm)
ground  = get_task("grounding", vlm=vlm, segmenter=seg)
vqa     = get_task("vqa", vlm=vlm, segmenter=seg)

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

## 100% Config-Driven Execution

```yaml
# my_config.yaml
models:
  vlm:
    name: "huggingface"
    model_id: "Qwen/Qwen3-VL-8B-Instruct"
    device: "cuda"
    torch_dtype: "auto"
  segmenter:
    name: "isro-sam3"
    model_id: "facebook/sam3"

tasks:
  - name: "captioning"
    enabled: true
    prompt_template: "default_caption"
  - name: "grounding"
    enabled: true
    prompt_template: "default_grounding_extraction"
  - name: "vqa"
    enabled: true
    numeric_prompt_template: "vqa_numeric"
    binary_prompt_template: "vqa_binary"
    semantic_prompt_template: "vqa_semantic"

dataset:
  name: "json_dataset"
  input_json: "query.json"
```

Then:
```bash
geonli-run --config my_config.yaml
```

## Switch Models Instantly

| Model | Config Change |
|-------|--------------|
| Qwen3-VL 8B | `name: "huggingface"` + `model_id: "Qwen/Qwen3-VL-8B-Instruct"` |
| LLaVA 1.6 | `name: "huggingface"` + `model_id: "liuhaotian/llava-v1.6-vicuna-7b"` |
| InternVL2 | `name: "huggingface"` + `model_id: "OpenGVLab/InternVL2-4B"` |
| Your LoRA | `name: "isro-qwen3-vl"` (adapter layer, keeps your existing code) |
| OpenAI GPT-4V | Implement `OpenAIVLM(VLMBase)`, register as `openai-gpt4v` |

## Switch Datasets Instantly

| Dataset | Config |
|---------|--------|
| GeoNLI JSON | `name: "json_dataset"`, `input_json: "queries.json"` |
| HuggingFace Hub | `name: "huggingface"`, `dataset_name_or_path: "nlphuji/flickr30k"` |
| CSV/TSV | `name: "csv_dataset"`, `csv_path: "data.csv"` |
| Image folder + JSON sidecars | `name: "image_folder"`, `image_dir: "./images"` |

## Prompt-First Design: Zero Hardcoded Strings

Every task prompt lives in `geonli/prompts/templates/`:

| Template File | Used By |
|--------------|---------|
| `default_caption.txt` | CaptioningTask system prompt |
| `default_grounding_extraction.txt` | GroundingTask target-class extraction |
| `default_router.txt` | VQA LLM router (SAM vs VLM) |
| `vqa_numeric.txt` | VQA numeric system prompt |
| `vqa_binary.txt` | VQA binary system prompt |
| `vqa_semantic.txt` | VQA semantic system prompt |
| `satellite_agent.txt` | Tool-calling agent system prompt + examples |

To customize: copy any file, edit it, and point to it in config:
```yaml
tasks:
  - name: "captioning"
    prompt_template: "my_custom_caption"
```

If a template is missing, the package **raises a clear error** — there are no hidden fallbacks.

## Architecture

| Layer | Responsibility |
|-------|---------------|
| `core` | Abstract contracts (`VLMBase`, `SegmenterBase`, `TaskBase`), Registry, Config |
| `models` | Concrete model backends (`DummyVLM`, `HuggingFaceVLM`, `ISRO_VLM`) |
| `tasks` | `CaptioningTask`, `GroundingTask`, `VQATask` + `TransformersSatelliteAgent` |
| `datasets` | `JsonGeoNLIDataset`, `HuggingFaceGeoNLIDataset`, `CSVGeoNLIDataset`, `ImageFolderGeoNLIDataset` |
| `prompts` | `PromptManager` loads `.txt` / `.yaml` / `.json` prompt banks |
| `adapters` | Bridges to existing monolithic code (`isro_geonli.py`) |
| `cli` | `geonli-run`, `geonli-eval` |

## Adding a Custom Model (3 steps)

```python
from geonli import VLMBase, register_vlm

@register_vlm("my-model")
class MyModel(VLMBase):
    def __init__(self, model_id: str, **kwargs):
        ...
    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0.0, **kwargs) -> str:
        ...
    def model_name(self) -> str:
        return "my-model"
```

Available in any config via `name: "my-model"`.

## License

MIT
