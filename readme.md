# ISRO-GeoNLI

A production-ready pipeline for remote sensing image analysis using Vision Language Models (VLM) and SAM3:
- **Image Captioning**: Generate descriptions of aerial/satellite imagery
- **Object Grounding**: Detect and localize objects with oriented bounding boxes
- **Visual Question Answering**: Answer numeric, binary, and semantic questions

**Model**: Fine-tuned Qwen3-VL-8B (`Dinosaur2314/qwen_finetune11`)

## Repository Structure

```
ISRO-GeoNLI/
├── app_prod.py              # Production FastAPI server (preloads models)
├── app_dev.py               # Development server (lazy loading)
├── rs_pipeline.py           # Main RSPipeline class
├── api_helpers.py           # API utilities
├── api_models.py            # Pydantic schemas
│
├── model/                   # Model interfaces
│   ├── model_builder.py     # VLM & SAM3 initialization
│   ├── vlm_interface.py     # Qwen3-VL interface
│   └── sam3_interface.py    # SAM3 interface
│
├── tasks/                   # Task handlers
│   ├── captioning.py        # Image captioning
│   ├── grounding.py         # Object detection
│   └── vqa.py               # Visual QA
│
├── utils/                   # Utilities
│   ├── geo_calc.py          # Geometric calculations
│   ├── visualization.py     # Annotation tools
│   └── vqa_output_normalizer.py
│
├── Evaluation/              # Evaluation scripts
├── Finetuning_runs/         # Training scripts
├── website-backend/         # Web API backend
└── website-frontend/        # React frontend
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- HuggingFace account with model access

### Installation

```bash
# Clone repository
git clone https://github.com/Vijayavallabh/ISRO-GeoNLI.git
cd ISRO-GeoNLI

# Create environment
conda create -n isro_geonli python=3.10
conda activate isro_geonli

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# HuggingFace authentication
huggingface-cli login
```

### Run Production Server

```bash
# Linux/Mac
./run_prod.sh

# Windows
uvicorn app_prod:app --host 0.0.0.0 --port 8080

# Development mode (with auto-reload)
uvicorn app_dev:app --host 0.0.0.0 --port 8000 --reload
```

The server preloads models on startup (takes 2-3 minutes) and runs at `http://localhost:8080`.

### Test API

```bash
# Health check
curl http://localhost:8080/health

# Simple query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the area of the building?", "image_url": "https://example.com/image.jpg"}'
```

## API Endpoints

### POST /process
Structured request matching `query.json` schema.

**Request:**
```json
{
  "input_image": {
    "image_id": "sample_001",
    "image_url": "https://example.com/image.jpg",
    "metadata": {"spatial_resolution_m": 1.57}
  },
  "queries": {
    "caption_query": {"instruction": "Describe the image."},
    "grounding_query": {"instruction": "Locate all buildings."},
    "attribute_query": {
      "binary": {"instruction": "Is there any aeroplane?"},
      "numeric": {"instruction": "What is the area?"},
      "semantic": {"instruction": "What color is the building?"}
    }
  }
}
```

### POST /query
Auto-classifies query type using LLM.

**Request:**
```json
{
  "query": "Count the cars in the parking lot",
  "image_url": "https://example.com/parking.jpg"
}
```

## Pipeline Architecture

### High-Level Task Flows

#### 1. Image Captioning
```
Input: Image + Instruction
         ↓
    VLM Interface (Qwen3-VL)
         ↓
    Generate Caption
         ↓
    Output: Caption String
```


#### 2. Object Grounding (Two-Stage)
```
Input: Image + Query ("Locate all airports")
         ↓
Stage 1: VLM Coarse Detection
  • Parse query → extract target classes
         ↓
Stage 2: SAM3 Call
  • SAM3 segmentation → precise masks
  • Extract oriented bounding boxes (OBB)
  • Format: [x1,y1, x2,y2, x3,y3, x4,y4]
         ↓
Post-Processing
  • De-duplication (distance-based filtering)
  • Area calculation (pixel_area × gsd²)
  • Assign unique object IDs
         ↓
Output: [{object-id, obbox, area_m²}, ...]
```

#### 3. Visual Question Answering (Smart Router)
```
Input: Image + Question
         ↓
    Question Type Classification
    (Binary / Numeric / Semantic)
         ↓
    ┌─────────┴──────────┐
    ↓                    ↓
Binary/Semantic      Numeric
(Direct VLM)        (Grounding-based)
    ↓                    ↓
    |              Auto-Grounding
    |              (if not provided)
    |                    ↓
    |              Extract Metadata
    |              (count, areas)
    |                    ↓
    |              VLM with Metadata
    └─────────┬──────────┘
              ↓
    Answer Normalization
    • Binary: "Yes"/"No"
    • Numeric: number + unit
    • Semantic: text
              ↓
    Output: Answer String
```

## Python API Usage

```python
from PIL import Image
from rs_pipeline import RSPipeline

# Initialize
pipeline = RSPipeline(
    vlm_model_id="Dinosaur2314/qwen_finetune11",
    sam_model_id="facebook/sam3"
)

image = Image.open("satellite.jpg")
gsd = 1.57  # Ground Sample Distance (meters/pixel)

# Captioning
caption = pipeline.generate_caption(image, "Describe this aerial image.")

# Grounding
detections = pipeline.ground_objects(image, "Locate all airports", score_threshold=0.4)

# VQA
answer = pipeline.answer_question(
    image, 
    "How many aircraft are visible?",
    question_type="numeric",
    gsd=gsd
)
```

## Configuration

### Model Selection
```python
# Fine-tuned model (default)
pipeline = RSPipeline(vlm_model_id="Dinosaur2314/qwen_finetune11")

# Base model
pipeline = RSPipeline(vlm_model_id="Qwen/Qwen3-VL-8B")
```

### Parameters
- **score_threshold**: Confidence threshold for grounding (default: 0.4)
- **gsd**: Ground Sample Distance in meters/pixel (required for area calculations)
- **question_type**: "binary", "numeric", or "semantic" for VQA

## Troubleshooting

### CUDA Out of Memory
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Use CPU fallback (slower)
pipeline = RSPipeline(device="cpu")
```

### Model Access Denied
```bash
# Login to HuggingFace
huggingface-cli login

# Request access to gated models
# Visit https://huggingface.co/facebook/sam3
```

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### Slow Inference
- Use `app_prod.py` (preloads models) instead of `app_dev.py`
- Reduce image resolution before processing
- Increase `score_threshold` for faster grounding

### No Grounding Detections
- Lower `score_threshold` (default: 0.4 → try 0.2)
- Verify query phrasing: "Locate all X" instead of "Show me X"
- Ensure image quality (RGB mode, size > 512x512)

## License

- Pipeline Code: MIT License
- Qwen3-VL: Apache 2.0
- SAM3: [SAM License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)

## Citation

```bibtex
@article{qwen3vl2024,
  title={Qwen3-VL: Towards Versatile Vision-Language Understanding},
  author={Qwen Team},
  year={2024}
}

@article{sam3,
  title={Segment Anything Model 3},
  author={Meta AI Research},
  year={2024}
}
```

## Contact

- Repository: [github.com/Vijayavallabh/ISRO-GeoNLI](https://github.com/Vijayavallabh/ISRO-GeoNLI)
- Issues: [GitHub Issues](https://github.com/Vijayavallabh/ISRO-GeoNLI/issues)

