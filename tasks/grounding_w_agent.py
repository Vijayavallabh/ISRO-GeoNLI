"""
Object grounding task implementation with agentic mask selection.
"""
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.grounding_agent import GroundingSelectionAgent

import logging

logger = logging.getLogger(__name__)


class GroundingTask:
    """Handles object detection and grounding with agentic selection."""
    
    def __init__(self, vlm_interface, sam3_interface):
        """
        Args:
            vlm_interface: VLMInterface instance
            sam3_interface: SAM3Interface instance
        """
        self.vlm = vlm_interface
        self.sam3 = sam3_interface
        
        # Initialize agentic selection agent
        self.selection_agent = GroundingSelectionAgent(
            vlm_model=vlm_interface.model,
            vlm_processor=vlm_interface.processor,
            device=vlm_interface.device
        )
    
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
            output_text = self.vlm.query(None, prompt, max_tokens=10)
            
            # Clean up response
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
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
            
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        
        obb = [float(coord) for point in box_points for coord in point]
        
        return obb, largest_contour

    def create_annotated_image(self, original_image, masks, sam_metadata):
        """Create an annotated image with segmentation masks, IDs, and grid lines."""
        try:
            image_np = np.array(original_image)
            img_h, img_w = image_np.shape[:2]
            
            overlay = image_np.copy()
            
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
            
            for idx, (mask, meta) in enumerate(zip(masks, sam_metadata)):
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_np = mask.astype(np.uint8)
                mask_np = (mask_np > 0.5).astype(np.uint8)
                
                color = colors[idx]
                colored_mask = np.zeros_like(image_np)
                colored_mask[mask_np == 1] = color
                
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)
                
                obb = meta["obb"]
                cx = int((obb[0] + obb[2] + obb[4] + obb[6]) / 4)
                cy = int((obb[1] + obb[3] + obb[5] + obb[7]) / 4)
                
                text = str(meta["mask_id"])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(overlay, text, (cx - text_w//2, cy + text_h//2), 
                           font, font_scale, (0, 0, 0), thickness)
            
            # Add grid lines
            vertical_grid_color = (255, 0, 255)
            horizontal_grid_color = (0, 255, 255)
            grid_thickness = 1
            num_grid_lines = 10
            
            for i in range(1, num_grid_lines):
                x = int(img_w * i / num_grid_lines)
                cv2.line(overlay, (x, 0), (x, img_h), vertical_grid_color, grid_thickness)
            
            for i in range(1, num_grid_lines):
                y = int(img_h * i / num_grid_lines)
                cv2.line(overlay, (0, y), (img_w, y), horizontal_grid_color, grid_thickness)
            
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
        
        response = self.vlm.query(image, prompt_text, max_tokens=100)
        logger.info(f"   [Fallback] Response: '{response}'")
        
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if len(numbers) < 4 or len(numbers) % 4 != 0:
            logger.info(f"    [Fallback] Could not parse a valid number of coordinates ({len(numbers)}).")
            return []
            
        obbs = []
        
        for i in range(0, len(numbers), 4):
            try:
                x_min, y_min, x_max, y_max = [float(n) for n in numbers[i:i+4]]
                
                x_min = max(0, min(img_w, x_min))
                y_min = max(0, min(img_h, y_min))
                x_max = max(0, min(img_w, x_max))
                y_max = max(0, min(img_h, y_max))

                if x_max <= x_min or y_max <= y_min:
                    logger.info(f"    [Fallback] Skipping invalid box dimensions: {x_min} {y_min} {x_max} {y_max}")
                    continue
                    
                obb = [
                    x_min, y_min,
                    x_max, y_min,
                    x_max, y_max,
                    x_min, y_max
                ]
                obbs.append(obb)
            except ValueError:
                logger.debug("    [Fallback] Error converting parsed string to float.")
                continue
                
        return obbs    

    def select_masks_agentic(self, image, query, candidate_obbs, sam_metadata, masks):
        """
        Use agentic tool-calling to select appropriate masks based on query.
        Returns a tuple: (list of selected OBBs, list of selected indices).
        """
        if not candidate_obbs:
            return [], []
        
        # Create annotated image
        annotated_image = self.create_annotated_image(image, masks, sam_metadata)
        
        img_w, img_h = image.size
        
        # Use the selection agent
        selected_mask_ids = self.selection_agent.select_masks(
            query=query,
            masks_metadata=sam_metadata,
            image_size=(img_w, img_h),
            annotated_image=annotated_image
        )
        
        # Convert mask IDs to OBBs and indices
        selected_obbs = []
        selected_indices = []
        
        for mask_id in selected_mask_ids:
            # Find the corresponding metadata
            meta = next((m for m in sam_metadata if m['mask_id'] == mask_id), None)
            if meta and mask_id < len(candidate_obbs):
                selected_obbs.append(candidate_obbs[mask_id])
                selected_indices.append(mask_id)
        
        if selected_obbs:
            logger.info(f"    [Agentic Selection] Selected Mask IDs: {selected_indices}")
            return selected_obbs, selected_indices
        
        # Fallback: largest area if agent failed
        logger.info("    [Agentic Selection] Agent returned empty - falling back to largest mask")
        if candidate_obbs:
            areas = [meta['area'] for meta in sam_metadata]
            selected_idx = int(np.argmax(areas))
            return [candidate_obbs[selected_idx]], [selected_idx]
        
        return [], []

    def ground_objects(self, image, query, show_visualization=True):
        """
        Perform complete grounding pipeline: Extraction -> SAM3 -> Agentic Selection/Fallback.
        Returns a list of object results formatted as [{"object-id": "1", "obbox": [...]}, ...].
        """
        logger.info(f"--- Task: Grounding (Query: '{query}') ---")
        
        # Stage 1: Extract Target Class
        target_class = self.extract_target_class(query)
        logger.info(f"   [Extraction] Target Class: '{target_class}'")
        
        # Stage 2: SAM3 Segmentation
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
            
            # Reprocess masks to get OBBs and Areas
            reprocessed_metadata = []
            valid_obbs = []
            valid_masks = []
            
            for idx, mask in enumerate(masks):
                obb, contour = self.get_obb_from_mask(mask)
                if obb is not None:
                    area = cv2.contourArea(contour) if contour is not None else 0
                    confidence = 0.0
                    if idx < len(sam_results.get("metadata", [])):
                        confidence = sam_results["metadata"][idx].get("confidence", 0.0)
                        
                    reprocessed_metadata.append({
                        "mask_id": len(valid_obbs),
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

        # Stage 3: Agentic Selection or Fallback
        final_obbs = []
        selected_indices = []
        method = ""
        
        if not sam_success:
            # Fallback: Qwen Direct
            final_obbs = self.qwen_direct_localization(image, query)
            method = "qwen_direct"
            if final_obbs:
                logger.info(f"    [Qwen Direct] Found {len(final_obbs)} objects.")
        else:
            # Agentic Selection
            final_obbs, selected_indices = self.select_masks_agentic(
                image, query, candidate_obbs, sam_metadata_fallback, masks
            )
            method = "sam3_agentic_select"
            
        # Format result
        results = []
        for i, obb in enumerate(final_obbs):
            if method == "sam3_agentic_select" and selected_indices:
                meta = next((m for m in sam_metadata_fallback if m['mask_id'] == selected_indices[i]), None)
                score = meta.get("confidence", 1.0) if meta else 1.0
                full_metadata = meta
            elif method == "qwen_direct":
                score = 1.0
                full_metadata = {
                    "method": "qwen_direct",
                    "obb": obb,
                    "confidence": 1.0,
                    "area": 0.0,
                    "mask_id": i + 1
                }
            else:
                score = 1.0
                full_metadata = {"method": "unknown"}

            results.append({
                "object-id": str(i + 1),
                "description": query,
                "target_class": target_class,
                "obbox": obb,
                "score": score,  
                "metadata": full_metadata,
                "method": method
            })
            
        # Visualization
        if show_visualization and results:
            vis_img = np.array(image)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            
            # Draw Predictions (Green)
            for result in results:
                pred_obb = result['obbox']
                obj_id = result['object-id']
                
                pts = np.array(pred_obb).reshape(-1, 2)
                pts = np.vstack((pts, pts[0]))
                plt.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=3, label=f'Prediction {obj_id}')
                
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                plt.text(cx, cy, f"P:{obj_id}", color='white', fontsize=12, 
                         bbox=dict(facecolor='green', alpha=0.5))
            
            # Plot non-selected candidates
            if sam_success and method == "sam3_agentic_select":
                for meta in sam_metadata_fallback:
                    if meta['mask_id'] not in selected_indices:
                        c_obb = meta['obb']
                        pts = np.array(c_obb).reshape(-1, 2)
                        pts = np.vstack((pts, pts[0]))
                        plt.plot(pts[:, 0], pts[:, 1], 'y--', linewidth=1, alpha=0.7)
                        
                        cx = np.mean(pts[:, 0])
                        cy = np.mean(pts[:, 1])
                        plt.text(cx, cy, str(meta['mask_id']), color='black', fontsize=8,
                                 bbox=dict(facecolor='yellow', alpha=0.4))
                        
            plt.title(f"Result: {query}\nMethod: {method} ({len(results)} object(s) found)")
            plt.axis('off')
            
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='g', lw=3),
                            Line2D([0], [0], color='y', linestyle='--', lw=1)]
            plt.legend(custom_lines, ['Final Bounding Box(es)', 'Unselected Candidate(s)'], loc='upper right')
            
            plt.show()
            
        # Final output structure
        return [{
            "object-id": r["object-id"],
            "obbox": r["obbox"]
        } for r in results]
