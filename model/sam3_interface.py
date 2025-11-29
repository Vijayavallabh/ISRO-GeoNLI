"""
Interface for SAM 3 segmentation model.
"""

import torch
import numpy as np
from utils.geo_calc import GeoCalculator


class SAM3Interface:
    """Handles SAM 3 segmentation operations."""
    
    def __init__(self, sam_model, sam_processor, device="cuda"):
        """
        Args:
            sam_model: The loaded SAM 3 model
            sam_processor: The SAM 3 processor
            device: Device to run inference on
        """
        self.model = sam_model
        self.processor = sam_processor
        self.device = device
    
    def segment_from_crop(self, cropped_image, target_label, threshold=0.4):
        """
        Perform segmentation on a cropped image region.
        
        Args:
            cropped_image: PIL Image (cropped region)
            target_label: Text description of target object
            threshold: Confidence threshold for detections
            
        Returns:
            dict with keys: 'masks', 'scores', 'best_mask', 'best_score'
            or None if no valid detections
        """
        inputs = self.processor(
            images=cropped_image, 
            text=[target_label],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
            outputs, 
            threshold=threshold,
            target_sizes=[(cropped_image.height, cropped_image.width)]
        )[0]
        
        masks = results["masks"]
        scores = results["scores"]
        
        if len(masks) == 0:
            return None
        
        # Pick the best mask (highest score)
        best_idx = torch.argmax(scores).item()
        best_mask = masks[best_idx].cpu().numpy()
        best_score = scores[best_idx].item()
        
        return {
            'masks': masks,
            'scores': scores,
            'best_mask': best_mask,
            'best_score': best_score
        }
    
    def refine_detection(self, image, hbb, target_label, 
                        expansion_factor=0.2, score_threshold=0.4, gsd=1.0):
        """
        Refine a coarse bounding box detection using SAM 3.
        
        Args:
            image: PIL Image (full image)
            hbb: Horizontal bounding box [x1, y1, x2, y2]
            target_label: Text description of target
            expansion_factor: How much to expand the crop box for context
            score_threshold: Minimum confidence threshold
            gsd: Ground Sample Distance (meters per pixel)
            
        Returns:
            dict with detection metadata or None
        """
        img_w, img_h = image.size
        
        # Expand box for context
        crop_box = self._expand_box(hbb, img_w, img_h, expansion_factor)
        cx1, cy1, cx2, cy2 = crop_box
        cropped_img = image.crop((cx1, cy1, cx2, cy2))
        
        # Run SAM 3 segmentation
        seg_result = self.segment_from_crop(cropped_img, target_label, score_threshold)
        
        if seg_result is None:
            # Fallback: Use original HBB as OBB
            print(f"[SAM3] No masks found for '{target_label}' - using HBB as fallback.")
            return self._hbb_to_detection(hbb, target_label, gsd)
        
        # Convert mask to OBB in crop coordinates
        calc = GeoCalculator(spatial_resolution_m=gsd)
        obb_crop = calc.get_obb_from_mask(seg_result['best_mask'])
        
        if not obb_crop:
            return self._hbb_to_detection(hbb, target_label, gsd)
        
        # Transform OBB back to original image coordinates
        (crop_center_x, crop_center_y), (obb_w, obb_h), angle = obb_crop
        orig_center_x = crop_center_x + cx1
        orig_center_y = crop_center_y + cy1
        obb_orig = ((orig_center_x, orig_center_y), (obb_w, obb_h), angle)
        
        # Calculate area
        area_m2 = calc.calculate_polygon_area(obb_orig)
        
        return {
            "obb": obb_orig,
            "score": seg_result['best_score'],
            "hbb": hbb,
            "center_point": (orig_center_x, orig_center_y),
            "area_m2": area_m2,
            "label": target_label,
            "is_fallback": False
        }
    
    def _expand_box(self, bbox, img_width, img_height, factor=0.2):
        """Expand bounding box by factor while staying in image bounds."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        dx, dy = w * factor, h * factor
        
        new_x1 = max(0, x1 - dx)
        new_y1 = max(0, y1 - dy)
        new_x2 = min(img_width, x2 + dx)
        new_y2 = min(img_height, y2 + dy)
        
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    
    def _hbb_to_detection(self, hbb, label, gsd):
        """Convert HBB to detection dict (fallback when SAM fails)."""
        x1, y1, x2, y2 = hbb
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        obb = ((cx, cy), (w, h), 0.0)
        calc = GeoCalculator(spatial_resolution_m=gsd)
        area_m2 = calc.calculate_polygon_area(obb)
        
        return {
            "obb": obb,
            "score": 0.99,  # High confidence to pass threshold
            "hbb": hbb,
            "center_point": (cx, cy),
            "area_m2": area_m2,
            "label": label,
            "is_fallback": True
        }
