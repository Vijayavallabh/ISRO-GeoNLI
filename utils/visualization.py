"""
Visualization utilities for drawing bounding boxes and annotations.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def annotate_image_with_boxes(image, obbs):
    """
    Draw oriented bounding boxes with IDs on image.

    Supports:
    - OpenCV RotatedRect: ((cx, cy), (w, h), angle)
    - Polygon OBB: [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    id_map = {}

    for idx, obb in enumerate(obbs, start=1):
        if obb is None:
            continue

        polygon = None
        cx, cy = None, None

        # --- Case 1: Polygon OBB (8 coords)
        if isinstance(obb, (list, tuple)) and len(obb) == 8:
            pts = np.array(obb, dtype=np.int32).reshape(4, 2)
            polygon = [tuple(pt) for pt in pts]
            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])

        # --- Case 2: OpenCV RotatedRect
        elif isinstance(obb, tuple) and len(obb) == 3:
            box_points = cv2.boxPoints(obb)
            box_points = np.int32(box_points)
            polygon = [tuple(pt) for pt in box_points]
            cx, cy = obb[0]

        else:
            raise ValueError(f"Unsupported OBB format: {obb}")

        # --- Draw polygon
        draw.polygon(polygon, outline="red", width=3)

        # --- Draw ID label
        text = str(idx)
        bbox = draw.textbbox((cx, cy), text, font=font)
        draw.rectangle(
            (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2),
            fill="white",
            outline="red"
        )
        draw.text((cx, cy), text, fill="black",
                 font=font, anchor="mm")

        id_map[idx] = obb

    return annotated_img, id_map

def draw_masks_on_image(image, masks, alpha=0.5):
    """
    Overlay segmentation masks on image.
    
    Args:
        image: PIL Image
        masks: List of binary masks (numpy arrays)
        alpha: Transparency factor
        
    Returns:
        PIL Image with mask overlays
    """
    import matplotlib.pyplot as plt
    
    img_array = np.array(image)
    overlay = img_array.copy()
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    for mask, color in zip(masks, colors):
        color_mask = (np.array(color[:3]) * 255).astype(np.uint8)
        overlay[mask] = overlay[mask] * (1 - alpha) + color_mask * alpha
    
    return Image.fromarray(overlay.astype(np.uint8))
