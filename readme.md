# ISRO-GeoNLI

A production-ready pipeline for remote sensing image analysis using Vision Language Models (VLM) and SAM3:
- **Image Captioning**: Generate descriptions of aerial/satellite imagery
- **Object Grounding**: Detect and localize objects with oriented bounding boxes
- **Visual Question Answering**: Answer numeric, binary, and semantic questions

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
- CUDA-capable GPU (36GB+ VRAM recommended)
- HuggingFace account with SAM 3 access

### Installation

```bash
# Clone repository
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
Structured request matching `query.json` schema. Please use this endpoint for evaluation

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
Auto-classifies query type using LLM. This is used internally in the chat app.

**Request:**
```json
{
  "query": "Count the cars in the parking lot",
  "image_url": "https://example.com/parking.jpg"
}
```
Please refer report (Team_46.pdf) for the architecture details.

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


