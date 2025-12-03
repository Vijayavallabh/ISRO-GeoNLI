"""
Object grounding task implementation.
"""
import re
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils.visualization import annotate_image_with_boxes


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
        
        print(f"\n[Grounding] Stage 1: Target Extraction for '{query}'")
        
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
            print(f"   [Extraction Warning] {e}")
            
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
            print(f"[Error] Failed to create annotated image: {e}")
            return original_image

    def qwen_direct_localization(self, image, description):
        """Fallback: Use VLM to directly predict horizontal bounding box."""
        img_w, img_h = image.size
        
        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n"
            f"TARGET DESCRIPTION: \"{description}\"\n"
            f"IMAGE SIZE: {img_w}x{img_h} pixels\n"
            f"NOTE: Coordinate (0,0) is at the top-left corner of the image.\n\n"
            f"TASK: Locate the object described above and provide a horizontal bounding box around it.\n"
            f"OUTPUT FORMAT: Provide 4 numbers: x_min y_min x_max y_max (top-left and bottom-right corners)\n"
            f"- x_min: left edge x-coordinate (0 to {img_w})\n"
            f"- y_min: top edge y-coordinate (0 to {img_h})\n"
            f"- x_max: right edge x-coordinate (0 to {img_w})\n"
            f"- y_max: bottom edge y-coordinate (0 to {img_h})\n\n"
            f"Example output: \"150 200 300 350\"\n\n"
            f"Respond with ONLY the 4 numbers separated by spaces, nothing else."
        )
        
        print("\n   [Fallback] SAM3 failed. Attempting Qwen Direct Localization...")
        
        # Pass actual image here
        response = self.vlm.query(image, prompt_text, max_tokens=50)
        print(f"   [Fallback] Response: '{response}'")
        
        # Parse coordinates
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if len(numbers) >= 4:
            x_min, y_min, x_max, y_max = [float(n) for n in numbers[:4]]
            
            # Clamp
            x_min = max(0, min(img_w, x_min))
            y_min = max(0, min(img_h, y_min))
            x_max = max(0, min(img_w, x_max))
            y_max = max(0, min(img_h, y_max))
            
            if x_max <= x_min or y_max <= y_min:
                print("   [Fallback] Invalid box dimensions.")
                return None
                
            # Convert to OBB (8 coords)
            obb = [
                x_min, y_min,  # top-left
                x_max, y_min,  # top-right
                x_max, y_max,  # bottom-right
                x_min, y_max   # bottom-left
            ]
            return obb
            
        print("   [Fallback] Could not parse 4 coordinates.")
        return None

    def select_best_obb(self, image, description, candidate_obbs, sam_metadata, masks):
        """Use VLM to select the best OBB from candidates using annotated image."""
        if not candidate_obbs:
            return None, -1
            
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
        
        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n"
            f"TARGET DESCRIPTION: \"{description}\"\n"
            f"CANDIDATE MASKS (shown with colored overlays and numeric IDs):\n"
            f"{obb_descriptions}\n"
            f"NOTE: The image has a 10x10 grid overlay to help with spatial reference. "
            f"Vertical lines are MAGENTA, horizontal lines are CYAN.\n"
            f"NOTE: Coordinate (0,0) is at the top-left corner of the image.\n"
            f"TASK: Identify which mask ID corresponds to the target object described above. "
            f"You are allowed to take time to reason and output the desired answer. \n"
            f"OUTPUT: Reply with ONLY the mask ID number (e.g., \"0\" or \"1\" or \"2\"). No explanation needed."
        )
        
        print("\n   [Selection] Asking VLM to select best candidate...")
        # Pass annotated image here
        response = self.vlm.query(annotated_image, prompt_text, max_tokens=20)
        print(f"   [Selection] Response: '{response}'")
        
        # Parse ID
        matches = re.findall(r'\d+', response)
        if matches:
            selected_idx = int(matches[0])
            if 0 <= selected_idx < len(candidate_obbs):
                print(f"   [Selection] Selected Mask ID: {selected_idx}")
                return candidate_obbs[selected_idx], selected_idx
        
        # Fallback: largest area
        print("   [Selection] Parsing failed. Selecting largest mask.")
        areas = [meta['area'] for meta in sam_metadata]
        selected_idx = int(np.argmax(areas))
        return candidate_obbs[selected_idx], selected_idx

    
    def ground_objects(self, image, query, show_visualization=True):
        """
        Perform complete grounding pipeline: Extraction -> SAM3 -> Selection/Fallback.
        """
        print(f"--- Task: Grounding (Query: '{query}') ---")
        
        # --- Stage 1: Extract Target Class ---
        target_class = self.extract_target_class(query)
        print(f"   [Extraction] Target Class: '{target_class}'")
        
        # --- Stage 2: SAM3 Segmentation ---
        sam_success = False
        masks = None
        sam_metadata_fallback = []
        candidate_obbs = []
        
        # Save PIL image to temp file for SAM3 interface
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
            
        try:
            sam_results = self.sam3.segment_image(temp_path, target_class)
        except Exception as e:
            print(f"   [SAM3 Error] {e}")
            sam_results = None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if sam_results and sam_results.get("masks") is not None and len(sam_results["masks"]) > 0:
            masks = sam_results["masks"]
            
            # Re-process masks to get reliable OBBs (8 coords) and Areas like the notebook
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
                print(f"   [SAM3] Found {len(candidate_obbs)} candidate masks.")
            else:
                print("   [SAM3] Found masks but failed to convert to OBBs.")
        else:
            print("   [SAM3] No masks found.")

        # --- Stage 3: Selection or Fallback ---
        final_obb = None
        selected_idx = -1
        method = "sam3_qwen_select"
        
        if not sam_success:
            # Fallback: Qwen Direct
            final_obb = self.qwen_direct_localization(image, query)
            method = "qwen_direct"
            if final_obb:
                sam_metadata_fallback = [{
                    "mask_id": -1,
                    "obb": final_obb,
                    "confidence": 0.0,
                    "area": 0.0,
                    "method": "qwen_direct"
                }]
                selected_idx = -1
        else:
            # Selection: Qwen Select
            final_obb, selected_idx = self.select_best_obb(
                image, query, candidate_obbs, sam_metadata_fallback, masks
            )
            
        # Format result
        if final_obb:
            result = {
                "id": 1,
                "description": query,
                "target_class": target_class,
                "obb": final_obb, # 8 coords
                "score": 1.0, 
                "metadata": sam_metadata_fallback,
                "selected_index": selected_idx,
                "method": method
            }
            results = [result]
        else:
            results = []
            
        # Visualization
        if show_visualization and results:
            pred_obb = results[0]['obb']
            
            # Prepare image for plotting
            vis_img = np.array(image)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            
            # Draw Prediction (Green)
            if pred_obb:
                pts = np.array(pred_obb).reshape(-1, 2)
                # Close the loop
                pts = np.vstack((pts, pts[0]))
                plt.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=3, label='Prediction')
                
                # Add label
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                plt.text(cx, cy, "PRED", color='white', fontsize=12, 
                         bbox=dict(facecolor='green', alpha=0.5))
            
            # If we have candidates (SAM succeeded), plot them faintly
            if sam_success:
                for meta in sam_metadata_fallback:
                    if meta['mask_id'] == selected_idx: continue
                    c_obb = meta['obb']
                    pts = np.array(c_obb).reshape(-1, 2)
                    pts = np.vstack((pts, pts[0]))
                    plt.plot(pts[:, 0], pts[:, 1], 'y--', linewidth=1, alpha=0.7)
                    
            plt.title(f"Result: {query}\nMethod: {method}")
            plt.axis('off')
            plt.legend()
            plt.show()
            
        return results