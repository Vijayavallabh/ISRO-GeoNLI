"""
Object grounding task implementation.
"""
import re
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


import logging

logger = logging.getLogger(__name__)


class GroundingTask:
    """Handles object detection and grounding."""
    
    def __init__(self, vlm_interface, sam3_interface):
        """
        Args:
            vlm_interface: VLMInterface instance
            sam3_interface: SAM3Interface instance
        """
        self.vlm = vlm_interface
        self.sam3 = sam3_interface
    
    def extract_target_class(self, query):
        """
        Extract the target object class from description using VLM.
        Uses text-only query mode (image=None).
        """
        prompt = (
            f"Extract only the main object type (1-2 words) from this description: '{query}'. "
            "Return ONLY the object type, nothing else."
        )
        
        logger.info(f"\n[Grounding] Stage 1: Target Extraction for '{query}'")
        
        try:
            # Pass None for image to use text-only mode
            output_text = self.vlm.query(None, prompt, max_tokens=10)
            
            # clean up response
            target_class = output_text.strip().lower()
            words = target_class.split()[:2]
            target_class = " ".join(words)
            
            if target_class:
                return target_class
                
        except Exception as e:
            logger.exception(f"   [Extraction Warning] {e}")
            
        # Fallback heuristic if VLM fails
        words = query.lower().split()
        skip_words = {'the', 'a', 'an', 'in', 'on', 'at', 'with', 'by', 'of', 'find', 'locate'}
        important_words = [w for w in words if w not in skip_words][:2]
        return " ".join(important_words) if important_words else "object"

    def get_obb_from_mask(self, mask):
        """Convert binary mask to oriented bounding box (8 coords)."""
        # Handle both torch tensor and numpy array
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
            
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        
        # Convert to x1y1x2y2x3y3x4y4 format (8 floats)
        obb = [float(coord) for point in box_points for coord in point]
        
        return obb, largest_contour

    def create_annotated_image(self, original_image, masks, sam_metadata):
        """Create an annotated image with segmentation masks, IDs, and grid lines."""
        try:
            # Convert PIL image to numpy array (RGB)
            image_np = np.array(original_image)
            img_h, img_w = image_np.shape[:2]
            
            # Create overlay image
            overlay = image_np.copy()
            
            # Generate distinct colors for each mask
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
            
            # Apply each mask with transparency
            for idx, (mask, meta) in enumerate(zip(masks, sam_metadata)):
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_np = mask.astype(np.uint8)
                mask_np = (mask_np > 0.5).astype(np.uint8)
                
                # Create colored mask
                color = colors[idx]
                colored_mask = np.zeros_like(image_np)
                colored_mask[mask_np == 1] = color
                
                # Blend with original image (30% opacity)
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)
                
                # Draw mask ID at the center of the OBB
                obb = meta["obb"]
                # Calculate center from x1y1...y4
                cx = int((obb[0] + obb[2] + obb[4] + obb[6]) / 4)
                cy = int((obb[1] + obb[3] + obb[5] + obb[7]) / 4)
                
                # Draw text
                text = str(meta["mask_id"])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                # Note: drawing on RGB image, color (0,0,0) is black
                cv2.putText(overlay, text, (cx - text_w//2, cy + text_h//2), font, font_scale, (0, 0, 0), thickness)
            
            # Add grid lines (10x10 grid)
            vertical_grid_color = (255, 0, 255)    # Magenta (RGB)
            horizontal_grid_color = (0, 255, 255)  # Cyan (RGB)
            grid_thickness = 1
            num_grid_lines = 10
            
            for i in range(1, num_grid_lines):
                x = int(img_w * i / num_grid_lines)
                cv2.line(overlay, (x, 0), (x, img_h), vertical_grid_color, grid_thickness)
            
            for i in range(1, num_grid_lines):
                y = int(img_h * i / num_grid_lines)
                cv2.line(overlay, (0, y), (img_w, y), horizontal_grid_color, grid_thickness)
            
            # Convert back to PIL Image
            return Image.fromarray(overlay)
            
        except Exception as e:
            logger.exception(f"[Error] Failed to create annotated image: {e}")
            return original_image

    def qwen_direct_localization(self, image, description):
        """
        Fallback: Use VLM to directly predict multiple horizontal bounding boxes.
        Returns a list of obb (8 coords) or an empty list.
        """
        img_w, img_h = image.size
        
        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n"
            f"TARGET DESCRIPTION: \"{description}\"\n"
            f"IMAGE SIZE: {img_w}x{img_h} pixels\n"
            f"NOTE: Coordinate (0,0) is at the top-left corner of the image.\n\n"
            f"TASK: Locate ALL objects described above and provide horizontal bounding boxes for them.\n"
            f"If you find multiple objects, list their coordinates one after another.\n"
            f"If no object is found, return an empty string.\n"
            f"OUTPUT FORMAT: Provide 4 numbers for EACH object: x_min y_min x_max y_max (top-left and bottom-right corners)\n"
            f"- x_min: left edge x-coordinate (0 to {img_w})\n"
            f"- y_min: top edge y-coordinate (0 to {img_h})\n"
            f"- x_max: right edge x-coordinate (0 to {img_w})\n"
            f"- y_max: bottom edge y-coordinate (0 to {img_h})\n\n"
            f"Example output for 2 objects: \"150 200 300 350 400 450 550 600\"\n\n"
            f"Respond with ONLY the numbers separated by spaces, nothing else."
        )
        
        logger.debug("\n   [Fallback] SAM3 failed. Attempting Qwen Direct Localization...")
        
        # Pass actual image here
        response = self.vlm.query(image, prompt_text, max_tokens=100)
        logger.info(f"   [Fallback] Response: '{response}'")
        
        # Parse coordinates
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        # Must have a count divisible by 4
        if len(numbers) < 4 or len(numbers) % 4 != 0:
            print(f"    [Fallback] Could not parse a valid number of coordinates ({len(numbers)}).")
            return []
            
        obbs = []
        
        for i in range(0, len(numbers), 4):
            try:
                x_min, y_min, x_max, y_max = [float(n) for n in numbers[i:i+4]]
                
                # Clamp and enforce x_min < x_max, y_min < y_max
                x_min = max(0, min(img_w, x_min))
                y_min = max(0, min(img_h, y_min))
                x_max = max(0, min(img_w, x_max))
                y_max = max(0, min(img_h, y_max))

                if x_max <= x_min or y_max <= y_min:
                    print(f"    [Fallback] Skipping invalid box dimensions: {x_min} {y_min} {x_max} {y_max}")
                    continue
                    
                # Convert to OBB (8 coords)
                obb = [
                    x_min, y_min,  # top-left
                    x_max, y_min,  # top-right
                    x_max, y_max,  # bottom-right
                    x_min, y_max   # bottom-left
                ]
                obbs.append(obb)
            except ValueError:
                logger.debug("    [Fallback] Error converting parsed string to float.")
                continue
                
        return obbs    

    def select_best_obbs(self, image, description, candidate_obbs, sam_metadata, masks):
        """
        Use VLM to select the best OBBs (plural) from candidates using annotated image.
        Returns a tuple: (list of selected OBBs, list of selected indices).
        """
        if not candidate_obbs:
            return [], []
            
        # Create annotated image
        annotated_image = self.create_annotated_image(image, masks, sam_metadata)
        
        # Create prompt
        obb_descriptions = "\n".join([
            f"Mask {meta['mask_id']}: corners=({meta['obb'][0]:.1f},{meta['obb'][1]:.1f}), "
            f"({meta['obb'][2]:.1f},{meta['obb'][3]:.1f}), "
            f"({meta['obb'][4]:.1f},{meta['obb'][5]:.1f}), "
            f"({meta['obb'][6]:.1f},{meta['obb'][7]:.1f})"
            for meta in sam_metadata
        ])
        
        all_ids = [str(meta['mask_id']) for meta in sam_metadata]
        
        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n"
            f"TARGET DESCRIPTION: \"{description}\"\n"
            f"CANDIDATE MASKS (shown with colored overlays and numeric IDs):\n"
            f"{obb_descriptions}\n"
            f"NOTE: The image has a 10x10 grid overlay to help with spatial reference. "
            f"Vertical lines are MAGENTA, horizontal lines are CYAN.\n"
            f"NOTE: Coordinate (0,0) is at the top-left corner of the image.\n"
            f"TASK: Identify ALL mask IDs that correspond to the target object(s) described above. "
            f"If no mask corresponds, return an empty string.\n"
            f"OUTPUT: Reply with ONLY the mask ID numbers (e.g., \"0 3 5 8\"). Use spaces to separate IDs. No explanation needed."
        )
        
        logger.info("\n   [Selection] Asking VLM to select best candidate...")
        # Pass annotated image here
        response = self.vlm.query(annotated_image, prompt_text, max_tokens=20)
        logger.info(f"   [Selection] Response: '{response}'")
        
        # Parse IDs
        matches = re.findall(r'\d+', response)
        
        selected_indices = []
        selected_obbs = []
        
        for match in matches:
            try:
                selected_idx = int(match)
                if 0 <= selected_idx < len(candidate_obbs):
                    if selected_idx not in selected_indices: # Ensure uniqueness
                        selected_indices.append(selected_idx)
                        selected_obbs.append(candidate_obbs[selected_idx])
            except ValueError:
                continue
                
        if selected_obbs:
            logger.info(f"    [Selection] Selected Mask IDs: {selected_indices}")
            return selected_obbs, selected_indices
            
        # Fallback: largest area (if VLM fails to select any)
        logging.info("    [Selection] Parsing failed or VLM selected none. Falling back to largest mask if multiple exist.")
        if candidate_obbs:
            areas = [meta['area'] for meta in sam_metadata]
            selected_idx = int(np.argmax(areas))
            return [candidate_obbs[selected_idx]], [selected_idx]
            
        return [], []

    
    def ground_objects(self, image, query, show_visualization=True):
        """
        Perform complete grounding pipeline: Extraction -> SAM3 -> Selection/Fallback.
        Returns a list of object results formatted as [{"object-id": "1", "obbox": [...]}, ...].
        """
        logger.info(f"--- Task: Grounding (Query: '{query}') ---")
        
        # --- Stage 1: Extract Target Class ---
        target_class = self.extract_target_class(query)
        logger.info(f"   [Extraction] Target Class: '{target_class}'")
        
        # --- Stage 2: SAM3 Segmentation ---
        sam_success = False
        masks = None
        sam_metadata_fallback = []
        candidate_obbs = []

        try:
            sam_results = self.sam3.segment_image(image, target_class)
        except Exception as e:
            logger.info(f"   [SAM3 Error] {e}")
            sam_results = None
        
        if sam_results and sam_results.get("masks") is not None and len(sam_results["masks"]) > 0:
            masks = sam_results["masks"]
            
            # Re-process masks to get reliable OBBs (8 coords) and Areas
            reprocessed_metadata = []
            valid_obbs = []
            valid_masks = []
            
            for idx, mask in enumerate(masks):
                obb, contour = self.get_obb_from_mask(mask)
                if obb is not None:
                    area = cv2.contourArea(contour) if contour is not None else 0
                    confidence = 0.0
                    # Try to retrieve score if available in sam_results metadata
                    if idx < len(sam_results.get("metadata", [])):
                        confidence = sam_results["metadata"][idx].get("confidence", 0.0)
                        
                    reprocessed_metadata.append({
                        "mask_id": len(valid_obbs), # Re-index for consistent 0..N
                        "obb": obb,
                        "confidence": confidence,
                        "area": float(area)
                    })
                    valid_obbs.append(obb)
                    valid_masks.append(mask)
            
            if valid_obbs:
                sam_success = True
                candidate_obbs = valid_obbs
                sam_metadata_fallback = reprocessed_metadata
                masks = valid_masks
                logger.info(f"   [SAM3] Found {len(candidate_obbs)} candidate masks.")
            else:
                logger.info("   [SAM3] Found masks but failed to convert to OBBs.")
        else:
            logger.info("   [SAM3] No masks found.")

        # --- Stage 3: Selection or Fallback ---
        final_obbs = []
        selected_indices = []
        method = ""
        
        if not sam_success:
            # Fallback: Qwen Direct (returns list of OBBs)
            final_obbs = self.qwen_direct_localization(image, query)
            method = "qwen_direct"
            if final_obbs:
                print(f"    [Qwen Direct] Found {len(final_obbs)} objects.")
        else:
            # Selection: Qwen Select (returns list of OBBs and indices)
            final_obbs, selected_indices = self.select_best_obbs(
                image, query, candidate_obbs, sam_metadata_fallback, masks
            )
            method = "sam3_qwen_select"
            
        # Format result
        results = []
        for i, obb in enumerate(final_obbs):
            # If SAM was used, use the specific metadata for the selected index
            if method == "sam3_qwen_select" and selected_indices:
                # Find metadata corresponding to the current OBB's index
                meta = next((m for m in sam_metadata_fallback if m['mask_id'] == selected_indices[i]), None)
                score = meta.get("confidence", 1.0) if meta else 1.0
                full_metadata = meta
                
            # If Qwen Direct was used
            elif method == "qwen_direct":
                # Create minimal metadata for Qwen direct output
                score = 1.0
                full_metadata = {
                    "method": "qwen_direct",
                    "obb": obb,
                    "confidence": 1.0,
                    "area": 0.0, # Cannot compute area easily here
                    "mask_id": i + 1
                }
            else:
                score = 1.0
                full_metadata = {"method": "unknown"}

            results.append({
                "object-id": str(i + 1), # 1-based indexing for output
                "description": query,
                "target_class": target_class,
                "obbox": obb, # 8 coords
                "score": score,  
                "metadata": full_metadata,
                "method": method
            })
            
        # Visualization
        if show_visualization and results:
            # Prepare image for plotting
            vis_img = np.array(image)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            
            # Draw Predictions (Green)
            for result in results:
                pred_obb = result['obbox']
                obj_id = result['object-id']
                
                pts = np.array(pred_obb).reshape(-1, 2)
                # Close the loop
                pts = np.vstack((pts, pts[0]))
                plt.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=3, label=f'Prediction {obj_id}')
                
                # Add label
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                plt.text(cx, cy, f"P:{obj_id}", color='white', fontsize=12, 
                         bbox=dict(facecolor='green', alpha=0.5))
            
            # If SAM succeeded, plot non-selected candidates faintly
            if sam_success and method == "sam3_qwen_select":
                for meta in sam_metadata_fallback:
                    if meta['mask_id'] not in selected_indices:
                        c_obb = meta['obb']
                        pts = np.array(c_obb).reshape(-1, 2)
                        pts = np.vstack((pts, pts[0]))
                        plt.plot(pts[:, 0], pts[:, 1], 'y--', linewidth=1, alpha=0.7)
                        # Add mask ID
                        cx = np.mean(pts[:, 0])
                        cy = np.mean(pts[:, 1])
                        plt.text(cx, cy, str(meta['mask_id']), color='black', fontsize=8,
                                 bbox=dict(facecolor='yellow', alpha=0.4))
                        
            plt.title(f"Result: {query}\nMethod: {method} ({len(results)} object(s) found)")
            plt.axis('off')
            
            # Create a simple legend for the visualization to clarify colors
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='g', lw=3),
                            Line2D([0], [0], color='y', linestyle='--', lw=1)]
            plt.legend(custom_lines, ['Final Bounding Box(es)', 'Unselected Candidate(s)'], loc='upper right')
            
            plt.show()
            
        # Final output structure matches the request
        return [{
            "object-id": r["object-id"],
            "obbox": r["obbox"]
        } for r in results]
