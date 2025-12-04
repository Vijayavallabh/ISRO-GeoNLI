import cv2
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Processor, Sam3Model # Native SAM 3 classes

def get_obb_from_mask(mask):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle (OBB)
    obb = cv2.minAreaRect(largest_contour)

    return obb, largest_contour

def run_sam3_text(image_path, text_prompt, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 on {device}...")

    # 1. Load Model and Processor explicitly
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image = Image.open(image_path).convert("RGB")
    h, w = image.height, image.width

    # 2. Process inputs with the text prompt
    # SAM 3 native text prompting requires passing 'text' to the processor
    print(f"Prompting SAM 3 with: '{text_prompt}'")
    inputs = processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt"
            ).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-process results
    # This handles resizing masks back to original image size and filtering by threshold
    results = processor.post_process_instance_segmentation(
            outputs, 
            threshold=0.4,  # Adjust confidence threshold
            target_sizes=[(h, w)]
            )[0]

    masks = results["masks"] # shape: (num_masks, H, W)
    scores = results["scores"]
    obbs = []
    print(f"Found {len(masks)} masks for '{text_prompt}'")
    for mask in masks:
        obb, contour = get_obb_from_mask(mask)
        if obb is not None:
            obbs.append(obb)
            center, (width, height), angle = obb
            print(f"  OBB - Center: {center}, Size: ({width:.1f}, {height:.1f}), Angle: {angle:.1f}°")

    dpi = 300
    figsize = (w / dpi, h / dpi)
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.imshow(np.array(image))
    ###################################### CAUTION : DO NOT RUN IF CPU IS BLOCKED MIGHT TAKE LONGER #################################
    '''
    if len(masks) > 0:
        # Convert tensor masks to numpy
        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy()

        # Sort by score
        sorted_indices = np.argsort(scores_np)[::-1]

        overlay = np.zeros((h, w, 4), dtype=np.float32)

        for idx in sorted_indices:
            mask = masks_np[idx]
            score = scores_np[idx]

            # Basic visual overlay
            color = np.random.random(3)
            alpha = 0.6
            color_mask = np.concatenate([color, [alpha]])
            overlay[mask] = color_mask
            print(f"Mask score: {score:.2f}")

        ax.imshow(overlay)
    '''
    for idx, obb in enumerate(obbs):
        if obb is not None:
            # Get OBB box points
            box_points = cv2.boxPoints(obb)
            box_points = np.int32(box_points)

            # Draw the OBB
            ax.plot(np.append(box_points[:, 0], box_points[0, 0]),
                    np.append(box_points[:, 1], box_points[0, 1]),
                    'r-', linewidth=2)

        # Draw center point
        center = obb[0]
        ax.plot(center[0], center[1], 'r*', markersize=15)

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=dpi)
    print(f"Saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--prompt", required=True) # e.g., "a red car"
    args = parser.parse_args()

    image_file = os.getcwd + args.in_file
    out_file = image_file.replace(".jpg", "_out.jpg").replace(".png", "_out.png")

    run_sam3_text(image_file, args.prompt, out_file)
