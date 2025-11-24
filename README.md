# Team_46

This is the repository of Team_46

## Environment Setup

This project uses `uv` for managing Python virtual environments and dependencies. `uv` is a fast, drop-in replacement for pip and virtualenv, built in Rust.

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Install `uv`:
   - On macOS and Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - On Windows:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
     ```
   - Or via pip (if you have Python installed):
     ```bash
     pip install uv
     ```

2. Clone the repository :
   ```bash
   git clone https://github.com/Vijayavallabh/ISRO-GeoNLI.git
   cd ISRO-GeoNLI
   git clone https://huggingface.co/datasets/xiang709/VRSBench
   cd VRSBench
   unzip Images_train.zip
   unzip Images_val.zip
   unzip Annotations_train.zip
   unzip ANnotations_val.zip
   ```

## Important
Navigate to VRSBench folder and then in Annotations_train folder, go to P0212_0000.json which has the last qa pair ques_id as "", change it to 5. Then go to Annotations_val folder, go to P1732_0000.json has first qa pair ques_id as "1", change it to integer.

### Create and Activate Virtual Environment

1. Create a virtual environment using `uv`:
   ```bash
   uv venv --python 3.11.9
   ```
   This creates a `.venv` directory in the project root.

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Windows (cmd):
     ```cmd
     .venv\Scripts\activate.bat
     ```

### Install Dependencies

- If the project has a `requirements.txt`:
  ```bash
  uv pip install -r requirements.txt
  ```
- If using a `pyproject.toml` (uv supports it):
  ```bash
  uv sync
  ```
- To add new dependencies:
  ```bash
  uv add package-name
  ```

## Evaluation Script (eval.py)

The [`eval.py`](eval.py ) script provides the `GeoNLIEvaluator` class for evaluating GeoNLI (Geospatial Natural Language Inference) tasks. It supports evaluation of captioning, grounding, binary, numeric, and semantic predictions using metrics like BERT-BLEU, IoU, cosine similarity, and more.

### Key Features
- Loads predictions and ground truths from JSON files.
- Computes weighted final scores for multiple tasks.
- Integrates with Weights & Biases (wandb) for logging.
- Handles coordinate conversions (pixels, meters, normalized).

### Usage
Run the script directly for example evaluation:
```bash
python eval.py
```
This loads a sample JSON file, evaluates predictions against themselves (for demonstration), and logs results to wandb.

To use in code:
```python
from eval import GeoNLIEvaluator

evaluator = GeoNLIEvaluator()
predictions, metadata = evaluator.load_predictions_from_json('path/to/response.json')
ground_truths = {...}  # Define ground truths
scores = evaluator.evaluate_all(predictions, ground_truths, metadata)
```

Adjust the command based on your project's entry point.

## Using the dataloader

This repository includes a lightweight streaming dataloader for the VRSBench dataset. It yields `VRSSample` objects with the following attributes:

- `image`: a `PIL.Image.Image` (RGB)
- `caption`: the image caption string
- `bbox`: a list of object bounding descriptions (format depends on dataset JSON)
- `bbox_questions`: list of referring-location prompts
- `vqa_questions`: list of VQA questions for the image
- `vqa_answers`: corresponding answers

Quick example to build dataloaders and inspect one sample:

```python
from dataloader import build_vrs_dataloaders

dls = build_vrs_dataloaders()

# Get a single sample from the test split
sample = next(iter(dls['test']))

print('Caption:', sample.caption)
print('Number of objects:', len(sample.bbox))
print('Image size (W,H):', sample.image.size)

# Use the PIL image size to set image dimensions for evaluators
width, height = sample.image.size
```

Note: the exact content/format of `sample.bbox` mirrors the dataset JSON (see `VRSBench/Annotations_*`). Adjust any conversion code depending on whether boxes are returned as pixel coords, normalized coords, or some custom schema.

## Quick grounding-evaluation test (example)

The following snippet shows a minimal grounding test using `GeoNLIEvaluator`. It demonstrates how to set image dimensions from a sample image and run `evaluate_grounding` with normalized boxes.

Create a small script `test_grounding.py` and run it with `python test_grounding.py`.

```python
from dataloader import build_vrs_dataloaders
from eval import GeoNLIEvaluator

# Build dataloaders and get one sample
dls = build_vrs_dataloaders()
sample = next(iter(dls['test']))

# Create evaluator and set image dims from the sample image
width, height = sample.image.size
evaluator = GeoNLIEvaluator(image_width=width, image_height=height)

# Example: if you have normalized oriented bounding boxes in the format
# [center_x, center_y, width, height, angle] where center & size are normalized (0..1)
# and angle is in degrees. This is a toy example where prediction equals ground truth.
pred_boxes = [ [0.5, 0.5, 0.2, 0.15, 0.0] ]
gt_boxes   = [ [0.51, 0.49, 0.19, 0.14, 0.0] ]

score = evaluator.evaluate_grounding(pred_boxes, gt_boxes,
                                     coordinate_system='normalized',
                                     validate_bounds=True)

print('Grounding score (normalized boxes):', score)
```

If you already have boxes in pixel coordinates, either pass `coordinate_system='meter'` (with `spatial_resolution_m`
set) and convert to pixels using the evaluator helpers, or pass absolute pixel boxes by converting them to the normalized
format shown above (divide x-coordinates by `image_width`, y-coordinates by `image_height`, etc.).

If you'd like, I can also add a tiny runnable example file (`test_grounding.py`) to the repo and a short CI-friendly unit test to exercise this path.

### Deactivating the Environment

To deactivate the virtual environment:

```bash
deactivate
```

### Additional Notes

- `uv` automatically handles dependency resolution and caching for faster installs.
- For more information, visit the [uv documentation](https://docs.astral.sh/uv/).
- If you encounter issues, ensure Python is installed and `uv` is up to date.
