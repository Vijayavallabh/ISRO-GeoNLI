# Remote Sensing Pipeline

A modular pipeline for remote sensing image analysis combining Vision Language Models (VLM) and SAM 3 for:
- **Image Captioning**: Generate detailed descriptions of aerial imagery
- **Object Grounding**: Detect and localize objects with oriented bounding boxes
- **Visual Question Answering**: Answer numeric, binary, and semantic questions

## Architecture

```
remote-sensing-pipeline/
├── model/
│   ├── model_builder.py      # Model initialization
│   ├── vlm_interface.py       # VLM query interface
│   └── sam3_interface.py      # SAM 3 segmentation interface
├── tasks/
│   ├── captioning.py          # Image captioning task
│   ├── grounding.py           # Object detection/grounding
│   └── vqa.py                 # Visual question answering
├── utils/
│   ├── geo_calculator.py      # Geometric calculations & GSD conversion
│   └── visualization.py       # Drawing utilities
├── examples/
│   └── example_usage.ipynb    # Example notebook
├── scripts/
│   └── run_from_json.py       # Run from JSON config
├── pipeline.py                # Main pipeline orchestrator
└── README.md
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Access to Qwen and SAM 3 model weights

### Setup

```bash
# Create environment
conda create -n rs_pipeline python=3.10
conda activate rs_pipeline

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install transformers accelerate pillow opencv-python matplotlib seaborn
pip install qwen-vl-utils  # For Qwen3-VL image processing
pip install -r requirements.txt

# Clone and install
git clone https://github.com/yourusername/remote-sensing-pipeline.git
cd remote-sensing-pipeline
```

## Quick Start

### Python API

```python
from PIL import Image
from pipeline import RSPipeline

# Initialize pipeline (uses Qwen3-VL-8B with transformers by default)
pipeline = RSPipeline(
    vlm_model_id="Qwen/Qwen3-VL-8B",
    sam_model_id="facebook/sam3"
)

# Load image
image = Image.open("aerial_image.jpg")
gsd = 1.57  # Ground Sample Distance in meters/pixel

# Task 1: Caption generation
caption = pipeline.generate_caption(
    image, 
    "Generate a detailed caption."
)

# Task 2: Object grounding
detections = pipeline.ground_objects(
    image, 
    "Locate the track field.",
    gsd=gsd
)

# Task 3: Visual QA
answer = pipeline.answer_question(
    image,
    "What is the area of the track field?",
    detections=detections,
    question_type="numeric",
    gsd=gsd
)
```

### JSON Configuration

Create a `query.json` file:

```json
{
  "input_image": {
    "image_id": "sample.png",
    "image_url": "https://example.com/image.jpg",
    "metadata": {
      "width": 512,
      "height": 512,
      "spatial_resolution_m": 1.57
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "Generate a detailed caption."
    },
    "grounding_query": {
      "instruction": "Locate the track field."
    },
    "attribute_query": {
      "binary": {"instruction": "Is there any aeroplane?"},
      "numeric": {"instruction": "What is the area of the track field?"},
      "semantic": {"instruction": "What is the color of the building?"}
    }
  }
}
```

Run the pipeline:

```bash
python scripts/run_from_json.py --input query.json --output results.json
```

## Features

### 1. Image Captioning
- **Generate-then-Compress Strategy**: Creates detailed draft then compresses to target length
- **High BLEU Scores**: Optimized for caption quality metrics
- Targets ~60 words while maintaining detail

### 2. Object Grounding
- **Two-Stage Pipeline**: 
  1. Coarse detection with VLM (full context)
  2. Fine-grained refinement with SAM 3 (cropped context)
- **Oriented Bounding Boxes**: Handles rotated objects
- **Real-world Measurements**: Automatic area calculation in m²
- **De-duplication**: Prevents duplicate detections

### 3. Visual Question Answering
- **Numeric Questions**: Area, distance, counting
- **Binary Questions**: Yes/No answers
- **Semantic Questions**: Color, material, activity
- **Dynamic Grounding**: Automatically detects missing objects
- **Metadata-Driven**: Uses detection metadata to reduce hallucination

## Pipeline Flow

```
Input Image + Query
        ↓
    [VLM] ← Extract target classes
        ↓
    Coarse Detection (HBB)
        ↓
    [SAM 3] ← Refine each box
        ↓
    Oriented Bounding Boxes (OBB)
        ↓
    De-duplication
        ↓
    Metadata Calculation
        ↓
    [VLM] ← Answer questions
        ↓
    Results
```

## Configuration

### Model Selection
- **VLM**: Default is Qwen3-VL-8B (uses transformers, no vLLM required)
- **SAM**: Uses facebook/sam3 (ensure HuggingFace access)

### GSD (Ground Sample Distance)
- Critical for accurate area/distance measurements
- Specify in meters per pixel
- Typically 0.5-2.0 m for aerial imagery

### Thresholds
- **SAM Score Threshold**: Default 0.4 (adjust for precision/recall trade-off)
- **De-duplication Distance**: Default 20 pixels

## Examples

See `examples/example_usage.ipynb` for comprehensive examples covering:
- Basic usage of all three tasks
- Working with different image sources
- Customizing parameters
- Interpreting results

## Troubleshooting

### CUDA Out of Memory
```python
pipeline = RSPipeline(gpu_memory_utilization=0.5)  # Reduce memory usage
```

### Model Loading Issues
- Ensure you have access to the model on HuggingFace
- Check that `trust_remote_code=True` is set (handled automatically)
- Verify transformers version: `pip install transformers>=4.40.0`

### SAM 3 Access Denied
Request access to the [SAM 3 HuggingFace repo](https://huggingface.co/facebook/sam3) and authenticate:
```bash
huggingface-cli login
```

## Citation

If you use this pipeline, please cite the underlying models:

```bibtex
@article{qwen3vl,
  title={Qwen3-VL: Towards Versatile Vision Language Models},
  author={Qwen Team},
  year={2024}
}

@article{sam3,
  title={Segment Anything Model 3},
  author={Meta AI},
  year={2024}
}
```

## License

This project follows the licenses of its dependencies:
- SAM 3: [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- Qwen: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
