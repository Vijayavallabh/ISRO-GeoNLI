# GeoNLI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **pip-installable**, **pluggable** toolkit for remote-sensing image analysis powered by Vision-Language Models (VLMs) and Segment Anything Models (SAM).

**Tasks:**
- **Image Captioning** — factual, detailed descriptions
- **Object Grounding** — oriented bounding boxes (OBB) with geometric features
- **Visual Question Answering (VQA)** — binary, numeric, and semantic questions

**Tested on:** NVIDIA A100 80GB with `Qwen/Qwen3-VL-8B-Instruct` + `facebook/sam3`

---

## Philosophy

| Principle | How GeoNLI Delivers |
|-----------|---------------------|
| **Any HuggingFace VLM** | One wrapper `HuggingFaceVLM` auto-detects Qwen / LLaVA / InternVL / Idefics families. Just provide a `model_id`. |
| **Any segmenter** | Built-in wrappers for SAM (vit-huge) and SAM3 (facebook/sam3). Add your own via `@register_segmenter`. |
| **Any dataset** | JSON, HuggingFace `datasets`, CSV/TSV, image-folder + sidecars — all via one config file. |
| **Any setup** | Local GPU, API-only, edge device — zero code changes. |
| **Every prompt is external** | All system prompts, router instructions, and agent examples live in `.txt` files. Customize behavior without touching Python. |

---

## Quick Start

### Installation

```bash
# Core package
pip install geonli

# With HuggingFace backends (recommended)
pip install geonli[transformers,sam]

# Full development stack
pip install geonli[all]
```

### Run Your First Experiment

```bash
# Run with built-in example config (uses dummy models for testing)
geonli-run --config geonli/configs/default.yaml

# Run with a real HF VLM + SAM3 (requires HF token for gated SAM3)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
geonli-run --config geonli/configs/example_hf.yaml --prompt-dir ./my_prompts/
```

### Programmatic Usage

```python
from geonli import get_vlm, get_segmenter, get_task
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline
from PIL import Image

# 1. Pick any registered VLM and segmenter
vlm = get_vlm("huggingface", model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda")
seg = get_segmenter("hf-sam3", model_id="facebook/sam3", device="cuda")

# 2. Build tasks
caption = get_task("captioning", vlm=vlm, prompt_template="default_caption")
ground  = get_task("grounding", vlm=vlm, segmenter=seg,
                   prompt_template="default_grounding_extraction",
                   fallback_to_vlm=True, show_visualization=False)
vqa     = get_task("vqa", vlm=vlm, segmenter=seg)

# 3. Assemble pipeline
pipeline = DefaultGeoNLIPipeline(tasks=[caption, ground, vqa])

# 4. Run on a PIL image
img = Image.open("satellite.png")
results = pipeline.run(
    image=img,
    queries={
        "caption_query": {"instruction": "Describe the image."},
        "grounding_query": {"instruction": "Locate all buildings."},
        "attribute_query": {
            "binary": {"instruction": "Is there water?"},
            "numeric": {"instruction": "How many cars?"},
            "semantic": {"instruction": "What color is the roof?"},
        }
    },
    metadata={"spatial_resolution_m": 1.57},
)

print(results["captioning"].response)
print(f"Found {len(results['grounding'].response)} objects")
```

---

## 100% Config-Driven Execution

A single YAML file defines everything:

```yaml
# my_config.yaml
experiment_name: "my_geonli_run"
device: "cuda"
seed: 42

models:
  vlm:
    name: "huggingface"
    model_id: "Qwen/Qwen3-VL-8B-Instruct"
    device: "cuda"
    torch_dtype: "auto"
  segmenter:
    name: "hf-sam3"
    model_id: "facebook/sam3"
    device: "cuda"
    spatial_resolution_m: 1.57

tasks:
  - name: "captioning"
    enabled: true
    prompt_template: "default_caption"
    max_tokens: 512
  - name: "grounding"
    enabled: true
    prompt_template: "default_grounding_extraction"
    fallback_to_vlm: true
    score_threshold: 0.4
  - name: "vqa"
    enabled: true
    numeric_prompt_template: "vqa_numeric"
    binary_prompt_template: "vqa_binary"
    semantic_prompt_template: "vqa_semantic"

dataset:
  name: "json_dataset"
  input_json: "query.json"
  image_root: "./images"

output:
  save_dir: "./outputs"
  format: "json"
  log_wandb: false
```

Run it:
```bash
export HF_TOKEN=your_token_here
geonli-run --config my_config.yaml
```

---

## Switch Models Instantly

| Model Family | Example `model_id` | Config |
|--------------|-------------------|--------|
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | `name: "huggingface"` |
| LLaVA | `liuhaotian/llava-v1.6-vicuna-7b` | `name: "huggingface"` |
| InternVL | `OpenGVLab/InternVL2-4B` | `name: "huggingface"` |
| Idefics | `HuggingFaceM4/idefics2-8b` | `name: "huggingface"` |
| Your fine-tuned LoRA | `Dinosaur2314/qwen_finetune11` | `name: "isro-qwen3-vl"` |
| OpenAI / Gemini | `gpt-4o-vision` / `gemini-1.5-flash` | Implement adapter (see below) |

| Segmenter | Example `model_id` | Config |
|-----------|-------------------|--------|
| SAM3 (gated) | `facebook/sam3` | `name: "hf-sam3"` |
| SAM-vit-huge | `facebook/sam-vit-huge` | `name: "hf-sam"` |
| SAM2 | `facebook/sam2-hiera-large` | `name: "hf-sam"` |
| Dummy (CI/testing) | — | `name: "dummy"` |

---

## Switch Datasets Instantly

| Dataset Format | Config |
|----------------|--------|
| GeoNLI JSON | `name: "json_dataset"`, `input_json: "queries.json"` |
| HuggingFace Hub | `name: "huggingface"`, `dataset_name_or_path: "nlphuji/flickr30k"` |
| CSV / TSV | `name: "csv_dataset"`, `csv_path: "data.csv"` |
| Image folder + JSON sidecars | `name: "image_folder"`, `image_dir: "./images"` |

---

## Prompt-First Design: Zero Hardcoded Strings

Every prompt lives in `geonli/prompts/templates/*.txt`:

| Template | Used By | Customizable? |
|----------|---------|---------------|
| `default_caption.txt` | Captioning system prompt | ✅ Yes |
| `default_grounding_extraction.txt` | Target-class extraction | ✅ Yes |
| `default_router.txt` | VQA LLM router (SAM vs VLM) | ✅ Yes |
| `vqa_numeric.txt` | Numeric VQA system prompt | ✅ Yes |
| `vqa_binary.txt` | Binary VQA system prompt | ✅ Yes |
| `vqa_semantic.txt` | Semantic VQA system prompt | ✅ Yes |
| `satellite_agent.txt` | Tool-calling agent instructions | ✅ Yes |

To customize:
1. Copy a file, e.g. `cp default_caption.txt my_caption.txt`
2. Edit `my_caption.txt`
3. Reference it in config:
   ```yaml
   tasks:
     - name: "captioning"
       prompt_template: "my_caption"
   ```

If a template is missing, the package **raises a clear `KeyError`** — there are no hidden Python fallbacks.

---

## Architecture

```
geonli/
├── core/
│   ├── base.py          # VLMBase, SegmenterBase, TaskBase, GeoNLIPipeline
│   ├── registry.py      # @register_vlm, @register_task, etc.
│   ├── config.py        # Pydantic ExperimentConfig
│   └── pipeline_impl.py # DefaultGeoNLIPipeline orchestrator
├── models/
│   ├── base.py          # DummyVLM, DummySegmenter
│   ├── huggingface.py   # HuggingFaceVLM (auto-detects Qwen/LLaVA/InternVL)
│   ├── huggingface_sam.py   # HuggingFaceSAM (SAM-vit-huge)
│   └── huggingface_sam3.py  # HuggingFaceSAM3 (facebook/sam3)
├── tasks/
│   ├── captioning.py    # CaptioningTask
│   ├── grounding.py     # GroundingTask (full OBB + geometric features + VLM selection)
│   ├── vqa.py           # VQATask (LLM router + agent)
│   └── agent.py         # TransformersSatelliteAgent (multi-step tool calling)
├── datasets/
│   ├── base.py               # JsonGeoNLIDataset
│   ├── huggingface_dataset.py
│   ├── csv_dataset.py
│   └── image_folder.py
├── prompts/
│   ├── manager.py       # PromptManager loads .txt/.yaml/.json banks
│   └── templates/       # All external prompt files
├── adapters/
│   └── isro_geonli.py   # Backward-compatible bridge to existing ISRO-GeoNLI code
└── cli/
    └── run.py           # geonli-run entry point
```

---

## Grounding Pipeline Explained

The `GroundingTask` runs a **full multi-stage pipeline**:

1. **Target Extraction** — VLM compresses free-form query to a noun phrase (e.g. `"the red car on the left"` → `"red car"`)
2. **Segmentation** — Segmenter (SAM3/SAM) produces masks from text prompt
3. **OBB + Geometry** — Each mask is converted to an 8-point oriented bounding box with:
   - Width, Height, Area
   - Aspect ratio, Angle, Compactness
   - Normalized coordinates (0–1000 range)
4. **VLM Selection** — Annotated image (colored masks + IDs + 10×10 grid) is shown to VLM, which selects the best matches
5. **Fallback** — If segmenter fails, VLM Direct Localization parses bounding box coordinates directly
6. **Visualization** (optional) — Matplotlib plot with green finals / yellow dashed candidates

**Tested result** with SAM3 + Qwen3-VL on A100:
```
Query: "Locate all trees."
→ SAM3 found 19 masks
→ VLM selected 4 best trees
→ Output: 4 Detection objects with real OBB coordinates
```

---

## VQA Pipeline Explained

The `VQATask` intelligently routes each question:

| Route | When | How |
|-------|------|-----|
| **VLM** | Visual-attribute questions (`color`, `type`, `terrain`) | Direct `.query()` with type-specific system prompt |
| **SAM + Agent** | Quantification questions (`how many`, `area`, `distance`) | Multi-step tool-calling agent with `detect_objects`, `get_object_info`, `measure_distance`, `calculate` |

**Agent loop example:**
```
User: "How many buildings are there?"
Step 1: {tool: detect_objects, arguments: {target_class: "building"}}
Observation: Detected 5 'building' object(s). Indices: 0 to 4.
Final Answer: 5
```

**Zero-object handling:**
```
User: "How many skyscrapers are there?"
Step 1: {tool: detect_objects, arguments: {target_class: "skyscraper"}}
Observation: No 'skyscraper' objects detected in the image.
Final Answer: 0
```

---

## Adding a Custom Model

Implement two methods + decorator = done.

```python
from geonli import VLMBase, register_vlm

@register_vlm("openai-gpt4v")
class OpenAIVLM(VLMBase):
    def __init__(self, api_key: str, model_id: str = "gpt-4o", **kwargs):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id

    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0.0, **kwargs) -> str:
        # Encode image, call API, return text
        ...

    def model_name(self) -> str:
        return self.model_id
```

Now use it in any config:
```yaml
models:
  vlm:
    name: "openai-gpt4v"
    api_key: "${OPENAI_API_KEY}"
```

---

## Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU inference)
- HuggingFace token (for gated models like `facebook/sam3`)

Optional extras:
```bash
pip install transformers accelerate opencv-python matplotlib
```

---

## Citation

If you use GeoNLI in your research, please cite:

```bibtex
@software{geonli2025,
  title={GeoNLI: A Flexible Toolkit for Remote-Sensing Captioning, Grounding, and VQA},
  author={ISRO-GeoNLI Team},
  year={2025},
  url={https://github.com/your-org/geonli}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
