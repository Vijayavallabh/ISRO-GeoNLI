#!/usr/bin/env python3
"""
SAM3 test with a query that EXISTS in the image ("tree"),
so we see the full pipeline: SAM3 -> OBB + geometry -> VLM selection.
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
print("SAM3 Test: Query = 'tree' (exists in image)")
print("=" * 70)

vlm = get_vlm("huggingface", model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda", torch_dtype="auto")
seg = get_segmenter("hf-sam3", model_id="facebook/sam3", device="cuda")
img = Image.open("sample_image.png").convert("RGB")

caption = get_task("captioning", vlm=vlm)
ground = get_task("grounding", vlm=vlm, segmenter=seg, prompt_template="default_grounding_extraction",
                  fallback_to_vlm=True, show_visualization=False)

pipeline = DefaultGeoNLIPipeline(tasks=[caption, ground])

queries = {
    "caption_query": {"instruction": "Describe this satellite image."},
    "grounding_query": {"instruction": "Locate all trees."},
}

start = time.time()
results = pipeline.run(image=img, queries=queries, metadata={"spatial_resolution_m": 1.57})

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
        for det in detections[:5]:
            print(f"   ID={det.object_id}, score={det.score:.3f}")
            print(f"        obbox={det.obbox}")
            if det.metadata:
                print(f"        meta={ {k:v for k,v in det.metadata.items() if k != 'geometric'} }")
    else:
        print(f"Response: {str(result.response)[:300]}")

print(f"\nTotal time: {time.time() - start:.1f}s")
