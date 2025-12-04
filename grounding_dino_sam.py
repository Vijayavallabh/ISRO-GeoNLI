import torch
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

parser = argparse.ArgumentParser()
parser.add_argument("--in_file")
parser.add_argument("--prompt")

args = parser.parse_args()

from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline

model = "IDEA-Research/grounding-dino-base"
device = "cuda"
processor = AutoProcessor.from_pretrained(model)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device)

# need to add pwd 
image_file = "/home/gopalks/shivanshu/vision_language/test/" + args.in_file
image = Image.open(image_file)

prompt = [[args.prompt]]

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

print("processed image")

results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,         # enables phrase-level matching
        text_labels=prompt,                 # optional, for explicit mapping
        threshold=0.25,                     # box confidence
        text_threshold=0.25,                # text-logit threshold
        target_sizes=[(image.height, image.width)],
        )[0]

boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
idx = scores.argmax().item() # what are scores and labels
box, score, label = boxes[idx], float(scores[idx]), labels[idx]

out_file = image_file.replace(".jpg", "_out.jpg").replace(".png", "_out.png")

sam = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=0 if torch.cuda.is_available() else -1)

h, w = image.height, image.width
img_array = np.array(image)

print(f"Image dimensions: {h} x {w}")

masks = []

for idx, (box, score) in enumerate(zip(boxes, scores)):
    x0, y0, x1, y1 = [int(v) for v in box.tolist()]

    # Ensure coordinates are within image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    print(f"Processing box {idx}: [{x0}, {y0}, {x1}, {y1}]")

 # Crop the image section
    cropped_image = Image.fromarray(img_array[y0:y1, x0:x1])

    # Get crop dimensions
    crop_h, crop_w = cropped_image.height, cropped_image.width

    # Run SAM on the cropped region with box coordinates relative to the crop
    # The box for the cropped image is the full crop area 
    out = sam(cropped_image, input_boxes=[[0, 0, crop_w, crop_h]]) # passing an np arry to the sam model IS THIS NOT RGB ? 

    for mask_idx, cropped_mask in enumerate(out["masks"]):
        cropped_mask_np = cropped_mask.cpu().numpy()
        print(f"  Cropped mask shape: {cropped_mask_np.shape}, True pixels: {cropped_mask_np.sum()}")

        # Create a global mask of the full image size
        global_mask = np.zeros((h, w), dtype=bool)

        # Place the cropped mask into the global mask at the correct position
        global_mask[y0:y1, x0:x1] = cropped_mask_np
        masks.append(global_mask)

print("mask and boxes are done" + "="*60)


dpi = 300
figsize = (w / dpi, h / dpi)

# Create figure
fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
ax.imshow(img_array)

# draw the masks
if len(masks) > 0:
    print(f"{len(masks)}")
    sorted_masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    for i, mask in enumerate(sorted_masks):
        color = np.random.random(3)
        alpha = 0.6
        color_mask = np.concatenate([color, [alpha]])
        overlay[mask] = color_mask
        print(f"Mask {i} : color={color}, pixels={mask.sum()}")

    ax.imshow(overlay)


# draw bounding boxes 
for box in boxes:
    x0, y0, x1, y1 = [int(v) for v in box.tolist()]
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        edgecolor="yellow", linewidth=1, fill=False
        ))

ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=dpi)
print(f"Saved to {out_file}")
plt.close()


