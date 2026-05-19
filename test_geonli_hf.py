#!/usr/bin/env python3
"""
End-to-end test of geonli with a real HuggingFace VLM (Qwen3-VL-8B)
on A100 GPU. Tests Captioning, Grounding, and VQA.
"""
import os
import sys
import time
from PIL import Image

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geonli import get_vlm, get_segmenter, get_task
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline
from geonli.prompts.manager import PromptManager
import geonli

# ---------------------------------------------------------------------------
# 0. Load prompts
# ---------------------------------------------------------------------------
builtin_prompt_dir = os.path.join(os.path.dirname(geonli.__file__), "prompts", "templates")
PromptManager(builtin_prompt_dir)

print("=" * 70)
print("GeoNLI End-to-End Test on A100")
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
print(f"   Model name: {vlm.model_name()}")

# ---------------------------------------------------------------------------
# 2. Load Segmenter (dummy, since SAM3 needs HF auth)
# ---------------------------------------------------------------------------
print("\n[2/5] Loading Segmenter: dummy (SAM3 would need auth token)")
segmenter = get_segmenter("dummy")
print(f"   Segmenter: {segmenter.model_name()}")

# ---------------------------------------------------------------------------
# 3. Load image
# ---------------------------------------------------------------------------
image_path = "sample_image.png"
if not os.path.exists(image_path):
    print(f"\n[!] {image_path} not found, creating a synthetic test image.")
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    img.save(image_path)
else:
    print(f"\n[3/5] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
print(f"   Image size: {img.size}")

# ---------------------------------------------------------------------------
# 4. Build tasks
# ---------------------------------------------------------------------------
print("\n[4/5] Building tasks ...")
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
print("   Tasks ready: captioning, grounding, vqa")

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
    print(f"Route/Model : {result.metadata.get('model', 'N/A')} | {result.metadata}")
    if task_name == "grounding":
        detections = result.response
        print(f"Objects found: {len(detections)}")
        for det in detections[:5]:
            print(f"   ID={det.object_id}, score={det.score:.3f}, obbox={det.obbox}")
    else:
        response_text = str(result.response)
        print(f"Response: {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

print("\n" + "=" * 70)
print(f"Total pipeline time: {elapsed:.1f}s")
print("=" * 70)
print("\nTest complete!")
