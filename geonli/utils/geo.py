"""
Lightweight geometric helpers that do NOT depend on the old codebase.
"""

from typing import List, Optional
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def mask_to_obb(mask) -> Optional[List[float]]:
    """
    Convert a binary mask to an 8-point OBB polygon.
    Works with numpy arrays or torch tensors.
    """
    if cv2 is None:
        return None

    if hasattr(mask, "cpu"):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    mask_np = (mask_np > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return [float(coord) for pt in box for coord in pt]
