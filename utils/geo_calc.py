"""
Geometric calculation utilities for remote sensing image analysis.
Handles pixel-to-meter conversions and area calculations.
"""

import numpy as np
import cv2


class GeoCalculator:
    """
    Handles geometric calculations and pixel-to-meter conversions 
    based on the provided spatial resolution (GSD).
    """
    def __init__(self, spatial_resolution_m=1.0):
        """
        Args:
            spatial_resolution_m: Ground Sample Distance in meters per pixel
        """
        self.gsd = spatial_resolution_m

    def pixel_to_meter(self, pixel_dist):
        """Convert pixel distance to meters."""
        return pixel_dist * self.gsd

    def pixel_area_to_meter_sq(self, pixel_area):
        """Convert pixel area to square meters."""
        return pixel_area * (self.gsd ** 2)

    def calculate_polygon_area(self, obb_or_mask):
        """
        Computes real-world area (m^2) from an OBB or Mask.
        
        Args:
            obb_or_mask: Either a tuple ((cx, cy), (w, h), angle) or a binary mask array
            
        Returns:
            Area in square meters
        """
        if isinstance(obb_or_mask, tuple):  # It's an OBB ((cx, cy), (w, h), ang)
            (cx, cy), (w, h), angle = obb_or_mask
            pixel_area = w * h
        else:  # It's a mask
            pixel_area = np.sum(obb_or_mask > 0)
            
        return self.pixel_area_to_meter_sq(pixel_area)

    def get_obb_from_mask(self, mask_bool):
        """
        Converts a binary mask (H, W) into an Oriented Bounding Box (OBB).
        
        Args:
            mask_bool: Binary mask as boolean numpy array
            
        Returns:
            OBB as ((cx, cy), (w, h), angle) or None if invalid
        """
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: 
            return None
        
        # Select largest contour to ignore small segmentation noise
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # Filter extremely small artifacts (e.g., < 10 pixels area)
        if cv2.contourArea(largest_cnt) < 10:
            return None
            
        # cv2.minAreaRect returns ((center_x, center_y), (width, height), angle)
        rect = cv2.minAreaRect(largest_cnt)
        return rect
