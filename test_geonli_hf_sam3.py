#!/usr/bin/env python3
"""
End-to-end test of geonli with:
  - Real HF VLM: Qwen3-VL-8B-Instruct
  - Real HF Segmenter: facebook/sam3 (SAM3)
Tests Captioning, Grounding (with real SAM3 text-prompted segmentation), and VQA.
"""
import os
import sys
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geonli import get_vlm, get_segmenter, get_task
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline
from geonli.prompts.manager import PromptManager
import geonli

builtin_prompt_dir = os.path.join(os.path.dirname(geonli.__file__), "prompts", "templates")
PromptManager(builtin_prompt_dir)

print("=" * 70)
print("GeoNLI End-to-End Test: Qwen3-VL + SAM3 on A100")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load VLM
# ---------------------------------------------------------------------------
print("\n[1/5] Loading VLM: Qwen/Qwen3-VL-8B-Instruct ...")
start = time.time()
vlm = get_vlm(
    "huggingface",
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="auto",
)
print(f"   VLM loaded in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 2. Load SAM3 segmenter
# ---------------------------------------------------------------------------
print("\n[2/5] Loading SAM3: facebook/sam3 ...")
print("   NOTE: SAM3 requires HuggingFace token with access rights.")
start = time.time()
try:
    segmenter = get_segmenter(
        "hf-sam3",
        model_id="facebook/sam3",
        device="cuda",
        spatial_resolution_m=1.57,
    )
    print(f"   SAM3 loaded in {time.time() - start:.1f}s")
except Exception as e:
    print(f"   ERROR loading SAM3: {e}")
    print("   Falling back to SAM-vit-huge ...")
    segmenter = get_segmenter(
        "hf-sam",
        model_id="facebook/sam-vit-huge",
        device="cuda",
        spatial_resolution_m=1.57,
    )
    print(f"   SAM loaded in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 3. Load image
# ---------------------------------------------------------------------------
image_path = "sample_image.png"
print(f"\n[3/5] Loading image: {image_path}")
img = Image.open(image_path).convert("RGB")
print(f"   Image size: {img.size}")

# ---------------------------------------------------------------------------
# 4. Build tasks
# ---------------------------------------------------------------------------
print("\n[4/5] Building tasks with REAL segmenter ...")
caption_task = get_task("captioning", vlm=vlm, prompt_template="default_caption")
ground_task = get_task(
    "grounding",
    vlm=vlm,
    segmenter=segmenter,
    prompt_template="default_grounding_extraction",
    fallback_to_vlm=True,
    show_visualization=False,
)
vqa_task = get_task("vqa", vlm=vlm, segmenter=segmenter)
print("   Tasks ready")

# ---------------------------------------------------------------------------
# 5. Run pipeline
# ---------------------------------------------------------------------------
print("\n[5/5] Running pipeline ...")
pipeline = DefaultGeoNLIPipeline(tasks=[caption_task, ground_task, vqa_task])

queries = {
    "caption_query": {"instruction": "Describe this satellite image in detail."},
    "grounding_query": {"instruction": "Locate all buildings."},
    "attribute_query": {
        "binary": {"instruction": "Is there any water body in the image?"},
        "numeric": {"instruction": "How many buildings can you see?"},
        "semantic": {"instruction": "What is the dominant terrain type?"},
    },
}

start = time.time()
results = pipeline.run(
    image=img,
    queries=queries,
    metadata={"spatial_resolution_m": 1.57},
)
elapsed = time.time() - start

# ---------------------------------------------------------------------------
# 6. Report results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for task_name, result in results.items():
    print(f"\n--- {task_name.upper()} ---")
    print(f"Query : {result.query}")
    print(f"Meta  : {result.metadata}")
    if task_name == "grounding":
        detections = result.response
        print(f"Objects found: {len(detections)}")
        for det in detections[:10]:
            print(f"   ID={det.object_id}, score={det.score:.3f}")
            print(f"        obbox={det.obbox}")
            if det.metadata:
                print(f"        meta={ {k:v for k,v in det.metadata.items() if k != 'geometric'} }")
    else:
        response_text = str(result.response)
        print(f"Response: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")

print("\n" + "=" * 70)
print(f"Total pipeline time: {elapsed:.1f}s")
print("=" * 70)
print("\nTest complete!")
