"""
Tool-based selection utilities for grounding pipeline.
Provides functions to filter and select masks based on spatial and attribute criteria.
"""

from typing import List, Dict, Union
import numpy as np


def filter_masks_by_region(
    masks_metadata: List[Dict],
    region: str,
    image_size: tuple
) -> List[int]:
    """
    Filter masks by spatial region in the image.
    
    Args:
        masks_metadata: List of mask metadata dicts (with 'obb' key)
        region: 'top', 'bottom', 'left', 'right', 'center', 'top-left', etc.
        image_size: (width, height) of image
    
    Returns:
        List of mask_ids that match the region criteria
    """
    img_w, img_h = image_size
    filtered_ids = []
    
    region = region.lower().replace("-", " ").replace("_", " ").strip()
    
    for meta in masks_metadata:
        obb = meta.get('obb', [])
        if len(obb) < 8:
            continue
        
        # Calculate centroid from OBB (8 coords: x1,y1,x2,y2,x3,y3,x4,y4)
        cx = (obb[0] + obb[2] + obb[4] + obb[6]) / 4
        cy = (obb[1] + obb[3] + obb[5] + obb[7]) / 4
        
        # Region checks
        is_top = cy < img_h / 2
        is_bottom = cy >= img_h / 2
        is_left = cx < img_w / 2
        is_right = cx >= img_w / 2
        
        match = False
        
        if region == "top":
            match = is_top
        elif region == "bottom":
            match = is_bottom
        elif region == "left":
            match = is_left
        elif region == "right":
            match = is_right
        elif "top" in region and "left" in region:
            match = is_top and is_left
        elif "top" in region and "right" in region:
            match = is_top and is_right
        elif "bottom" in region and "left" in region:
            match = is_bottom and is_left
        elif "bottom" in region and "right" in region:
            match = is_bottom and is_right
        elif region == "center":
            match = (img_w * 0.25 < cx < img_w * 0.75) and \
                    (img_h * 0.25 < cy < img_h * 0.75)
        
        if match:
            filtered_ids.append(meta['mask_id'])
    
    return filtered_ids


def select_masks_by_attribute_rank(
    masks_metadata: List[Dict],
    attribute: str,
    rank_index: int
) -> List[int]:
    """
    Select mask(s) by ranking on an attribute.
    
    Args:
        masks_metadata: List of mask metadata
        attribute: Attribute to sort by ('area', 'confidence')
        rank_index: 1=smallest, -1=largest, 2=second smallest, etc.
    
    Returns:
        List containing single mask_id (or empty if invalid)
    """
    if attribute not in ['area', 'confidence']:
        return []
    
    valid_masks = [m for m in masks_metadata if attribute in m]
    
    if not valid_masks:
        return []
    
    sorted_masks = sorted(valid_masks, key=lambda x: x[attribute])
    
    try:
        if rank_index > 0:
            idx = rank_index - 1
        elif rank_index < 0:
            idx = rank_index
        else:
            return []
        
        selected = sorted_masks[idx]
        return [selected['mask_id']]
        
    except IndexError:
        return []


def select_masks_by_attribute_threshold(
    masks_metadata: List[Dict],
    attribute: str,
    threshold: float,
    comparison: str = "greater"
) -> List[int]:
    """
    Select masks that meet an attribute threshold.
    
    Args:
        masks_metadata: List of mask metadata
        attribute: Attribute to filter by ('area', 'confidence')
        threshold: Threshold value
        comparison: 'greater', 'less', 'equal'
    
    Returns:
        List of mask_ids that meet the criteria
    """
    if attribute not in ['area', 'confidence']:
        return []
    
    selected_ids = []
    
    for meta in masks_metadata:
        value = meta.get(attribute)
        if value is None:
            continue
        
        match = False
        if comparison == "greater":
            match = value > threshold
        elif comparison == "less":
            match = value < threshold
        elif comparison == "equal":
            match = abs(value - threshold) < 0.01  # Float comparison
        
        if match:
            selected_ids.append(meta['mask_id'])
    
    return selected_ids


def select_all_masks(masks_metadata: List[Dict]) -> List[int]:
    """
    Select all available masks.
    
    Args:
        masks_metadata: List of mask metadata
    
    Returns:
        List of all mask_ids
    """
    return [meta['mask_id'] for meta in masks_metadata]


def select_masks_by_count(
    masks_metadata: List[Dict],
    count: int,
    attribute: str = "area",
    order: str = "largest"
) -> List[int]:
    """
    Select top N masks by an attribute.
    
    Args:
        masks_metadata: List of mask metadata
        count: Number of masks to select
        attribute: Attribute to sort by ('area', 'confidence')
        order: 'largest' or 'smallest'
    
    Returns:
        List of mask_ids for top N masks
    """
    if attribute not in ['area', 'confidence']:
        return []
    
    valid_masks = [m for m in masks_metadata if attribute in m]
    
    if not valid_masks:
        return []
    
    sorted_masks = sorted(valid_masks, key=lambda x: x[attribute], 
                         reverse=(order == "largest"))
    
    selected = sorted_masks[:min(count, len(sorted_masks))]
    return [m['mask_id'] for m in selected]


def calculate_mask_distance(
    masks_metadata: List[Dict],
    mask_id_1: int,
    mask_id_2: int
) -> Union[float, str]:
    """
    Calculate distance between centroids of two masks.
    
    Args:
        masks_metadata: List of mask metadata
        mask_id_1: First mask ID
        mask_id_2: Second mask ID
    
    Returns:
        Distance in pixels, or error string
    """
    mask1 = next((m for m in masks_metadata if m['mask_id'] == mask_id_1), None)
    mask2 = next((m for m in masks_metadata if m['mask_id'] == mask_id_2), None)
    
    if not mask1 or not mask2:
        return "Invalid mask ID(s)"
    
    obb1 = mask1.get('obb', [])
    obb2 = mask2.get('obb', [])
    
    if len(obb1) < 8 or len(obb2) < 8:
        return "Missing coordinate data"
    
    # Calculate centroids
    cx1 = (obb1[0] + obb1[2] + obb1[4] + obb1[6]) / 4
    cy1 = (obb1[1] + obb1[3] + obb1[5] + obb1[7]) / 4
    
    cx2 = (obb2[0] + obb2[2] + obb2[4] + obb2[6]) / 4
    cy2 = (obb2[1] + obb2[3] + obb2[5] + obb2[7]) / 4
    
    # Euclidean distance
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    return float(dist)


def get_mask_attribute(
    masks_metadata: List[Dict],
    mask_id: int,
    attribute: str
) -> Union[float, str, None]:
    """
    Get a specific attribute value from a mask.
    
    Args:
        masks_metadata: List of mask metadata
        mask_id: Mask ID to query
        attribute: Attribute to retrieve ('area', 'confidence', 'obb')
    
    Returns:
        Attribute value or None if not found
    """
    mask = next((m for m in masks_metadata if m['mask_id'] == mask_id), None)
    
    if not mask:
        return None
    
    return mask.get(attribute)
