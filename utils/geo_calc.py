"""
Geometric calculation utilities for remote sensing image analysis.
Handles pixel-to-meter conversions and area calculations.
"""

import numpy as np
import cv2


class GeoCalculator:

    def __init__(self, spatial_resolution_m=1.0):
        self.gsd = spatial_resolution_m

    def pixel_to_meter(self, pixel_dist):
        return pixel_dist * self.gsd

    def pixel_area_to_meter_sq(self, pixel_area):
        return pixel_area * (self.gsd ** 2)

    def extract_metadata_from_mask(self, mask_bool, gsd = None):
        current_gsd = gsd if gsd is not None else self.gsd
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_cnt = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_cnt) < 10:
            return None
        
        rect = cv2.minAreaRect(largest_cnt)
        box_points = cv2.boxPoints(rect)
        
        coordinates = [float(coord) for point in box_points for coord in point]
        
        epsilon = 0.02 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        num_vertices = len(approx)
        
        shape_map = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon"}
        shape = shape_map.get(num_vertices, "circle" if num_vertices > 6 else "irregular")

        pixel_area = np.sum(mask_bool > 0)
        area_m2 = pixel_area * (current_gsd ** 2)
        
        (cx, cy), (w, h), angle = rect
        width_m = max(w, h) * current_gsd
        height_m = min(w, h) * current_gsd
        
        orientation = float(angle) if angle >= 0 else float(angle + 180)
        
        return {
            "coordinates": coordinates,
            "shape": shape,
            "area_m2": float(area_m2),
            "width_m": float(width_m),
            "height_m": float(height_m),
            "orientation_deg": orientation
        }
