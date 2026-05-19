"""
Generic grounding task implementing the FULL original ISRO-GeoNLI pipeline:
  1. VLM extracts a clean noun phrase (Remote Sensing Specialist prompt)
  2. Segmenter returns masks
  3. Re-process masks -> OBBs (8 coords) + geometric features
  4. Normalize coordinates to 0-1000 range
  5. Create annotated image (colored masks + IDs + 10x10 grid)
  6. VLM selects best OBBs from candidates using annotated image
  7. Fallback to Qwen Direct Localization if segmenter fails
  8. Optional matplotlib visualization
Works with any VLMBase + SegmenterBase combination.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from geonli.core.base import (
    TaskBase,
    TaskResult,
    VLMBase,
    SegmenterBase,
    Detection,
)
from geonli.core.registry import register_task, get_prompt

logger = logging.getLogger(__name__)

# Soft-import CV2 / NumPy / Matplotlib so the module loads even without them
_cv2 = None
_np = None
_plt = None


def _import_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _import_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _import_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


@register_task("grounding")
class GroundingTask(TaskBase):
    name = "grounding"

    def __init__(
        self,
        vlm: VLMBase,
        segmenter: SegmenterBase,
        prompt_template: str = "default_grounding_extraction",
        max_tokens: int = 30,
        fallback_to_vlm: bool = True,
        score_threshold: float = 0.4,
        show_visualization: bool = False,
    ):
        self.vlm = vlm
        self.segmenter = segmenter
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.fallback_to_vlm = fallback_to_vlm
        self.score_threshold = score_threshold
        self.show_visualization = show_visualization

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        logger.info(f"--- Task: Grounding (Query: '{query}') ---")

        # --- Stage 1: Extract Target Class --------------------------------
        target_class = self._extract_target_class(query)
        logger.info(f"   [Extraction] Target Class: '{target_class}'")

        # --- Stage 2: Segment ---------------------------------------------
        sam_success = False
        masks = None
        sam_metadata_fallback: List[Dict[str, Any]] = []
        candidate_obbs: List[List[float]] = []

        try:
            sam_results = self.segmenter.segment(
                image, target_class, score_threshold=self.score_threshold
            )
        except Exception as e:
            logger.info(f"   [Segmenter Error] {e}")
            sam_results = None

        if sam_results and sam_results.masks and sam_results.count > 0:
            masks = sam_results.masks
            reprocessed_metadata = []
            valid_obbs = []
            valid_masks = []

            img_w, img_h = image.size
            img_area = img_w * img_h

            for idx, mask in enumerate(masks):
                obb, contour = self._get_obb_from_mask(mask)
                if obb is not None:
                    geom_features = self._extract_geometric_features(mask, obb)
                    if geom_features is None:
                        continue

                    area = geom_features["area"]
                    confidence = 0.0
                    if idx < len(sam_results.metadata):
                        confidence = sam_results.metadata[idx].get("confidence", 0.0)

                    reprocessed_metadata.append({
                        "mask_id": len(valid_obbs),
                        "obb": obb,
                        "confidence": confidence,
                        "area": area,
                        "geometric": {
                            "width_rel": geom_features["width"] / img_w,
                            "height_rel": geom_features["height"] / img_h,
                            "area_rel": area / img_area,
                            "aspect_ratio": geom_features["aspect_ratio"],
                            "angle": geom_features["angle"],
                            "is_horizontal": geom_features["is_horizontal"],
                            "compactness": geom_features["compactness"],
                        }
                    })
                    valid_obbs.append(obb)
                    valid_masks.append(mask)

            if valid_obbs:
                sam_success = True
                candidate_obbs = valid_obbs
                sam_metadata_fallback = reprocessed_metadata
                masks = valid_masks
                logger.info(f"   [Segmenter] Found {len(candidate_obbs)} candidate masks.")
            else:
                logger.info("   [Segmenter] Found masks but failed to convert to OBBs.")
        else:
            logger.info("   [Segmenter] No masks found.")

        # --- Stage 3: Selection or Fallback -----------------------------
        final_obbs: List[List[float]] = []
        selected_indices: List[int] = []
        method = ""

        if not sam_success:
            if self.fallback_to_vlm:
                final_obbs = self._qwen_direct_localization(image, query)
                method = "vlm_direct"
                if final_obbs:
                    logger.info(f"    [VLM Direct] Found {len(final_obbs)} objects.")
        else:
            final_obbs, selected_indices = self._select_best_obbs(
                image, query, candidate_obbs, sam_metadata_fallback, masks
            )
            method = "segmenter_vlm_select"

        # --- Format result ----------------------------------------------
        detections: List[Detection] = []
        for i, obb in enumerate(final_obbs):
            if method == "segmenter_vlm_select" and selected_indices:
                meta = next(
                    (m for m in sam_metadata_fallback if m["mask_id"] == selected_indices[i]),
                    None,
                )
                score = meta.get("confidence", 1.0) if meta else 1.0
                full_metadata = meta or {}
            elif method == "vlm_direct":
                score = 1.0
                full_metadata = {"method": "vlm_direct", "obb": obb, "confidence": 1.0, "area": 0.0, "mask_id": i}
            else:
                score = 1.0
                full_metadata = {"method": "unknown"}

            detections.append(
                Detection(
                    object_id=str(i + 1),
                    obbox=obb,
                    score=score,
                    metadata=full_metadata,
                )
            )

        # --- Visualization ----------------------------------------------
        if self.show_visualization and detections:
            self._visualize(image, detections, sam_success, sam_metadata_fallback, selected_indices, query, method)

        return TaskResult(
            task_name=self.name,
            query=query,
            response=detections,
            metadata={
                "model": self.segmenter.model_name(),
                "target_class": target_class,
                "method": method,
                "count": len(detections),
            },
        )

    # ------------------------------------------------------------------
    # Stage 1: Target class extraction
    # ------------------------------------------------------------------

    def _extract_target_class(self, query: str) -> str:
        logger.info(f"\n[Grounding] Stage 1: Target Extraction for '{query}'")
        try:
            try:
                prompt = get_prompt(self.prompt_template).format(query=query)
            except KeyError:
                prompt = self._default_extraction_prompt(query)

            output_text = self.vlm.query(
                image=None,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            target_class = output_text.strip().lower()
            target_class = target_class.strip('"\'.,;:')
            words = target_class.split()[:8]
            target_class = " ".join(words)
            if target_class:
                logger.info(f"   [Extraction] Target Class: '{target_class}'")
                return target_class
        except Exception as e:
            logger.exception(f"   [Extraction Warning] {e}")

        # Fallback heuristic
        words = query.lower().split()
        filler = {'the', 'a', 'an', 'this', 'that', 'is', 'are', 'in', 'on', 'at', 'of'}
        important = [w for w in words if w not in filler][:6]
        fallback = " ".join(important) if important else "object"
        logger.info(f"   [Extraction Fallback] Using: '{fallback}'")
        return fallback

    def _default_extraction_prompt(self, query: str) -> str:
        return (
            f'You are a Remote Sensing Segmentation Specialist.\n'
            f'Convert the user description into a "Simple Noun Phrase".\n'
            f'REMOVE spatial words, articles, and verbs.\n'
            f'USER DESCRIPTION: "{query}"\n'
            f'OUTPUT (Return ONLY the noun phrase):'
        )

    # ------------------------------------------------------------------
    # Stage 2: Mask -> OBB + geometric features
    # ------------------------------------------------------------------

    def _get_obb_from_mask(self, mask) -> Tuple[Optional[List[float]], Any]:
        cv2 = _import_cv2()
        np = _import_np()

        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = np.asarray(mask).astype(np.uint8)
        mask_np = (mask_np > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        obb = [float(coord) for point in box for coord in point]
        return obb, largest

    def _extract_geometric_features(self, mask, obb) -> Optional[Dict[str, Any]]:
        cv2 = _import_cv2()
        np = _import_np()

        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = np.asarray(mask).astype(np.uint8)
        mask_np = (mask_np > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        rect = cv2.minAreaRect(largest)
        (cx, cy), (w, h), angle = rect

        if w < h:
            w, h = h, w
            angle = (angle + 90) % 180

        aspect_ratio = w / (h + 1e-6)
        angle_norm = abs(angle) % 90
        is_horizontal = angle_norm < 5 or angle_norm > 85

        perimeter = cv2.arcLength(largest, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6) if perimeter > 0 else 0

        return {
            "width": float(w),
            "height": float(h),
            "area": float(area),
            "aspect_ratio": float(aspect_ratio),
            "angle": float(angle_norm),
            "is_horizontal": bool(is_horizontal),
            "compactness": float(compactness),
        }

    def _normalize_obb_to_1000(self, obb: List[float], img_w: int, img_h: int) -> List[float]:
        normalized = []
        for i in range(0, 8, 2):
            x_norm = (obb[i] / img_w) * 1000
            y_norm = (obb[i + 1] / img_h) * 1000
            normalized.extend([x_norm, y_norm])
        return normalized

    # ------------------------------------------------------------------
    # Stage 3a: VLM-based selection with annotated image
    # ------------------------------------------------------------------

    def _create_annotated_image(self, original_image: Image.Image, masks, sam_metadata) -> Image.Image:
        cv2 = _import_cv2()
        np = _import_np()
        try:
            image_np = np.array(original_image)
            img_h, img_w = image_np.shape[:2]
            overlay = image_np.copy()

            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

            for idx, (mask, meta) in enumerate(zip(masks, sam_metadata)):
                if hasattr(mask, "cpu"):
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_np = np.asarray(mask).astype(np.uint8)
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
                cv2.putText(overlay, text, (cx - text_w // 2, cy + text_h // 2), font, font_scale, (0, 0, 0), thickness)

            # 10x10 grid
            v_color = (255, 0, 255)
            h_color = (0, 255, 255)
            for i in range(1, 10):
                x = int(img_w * i / 10)
                cv2.line(overlay, (x, 0), (x, img_h), v_color, 1)
                y = int(img_h * i / 10)
                cv2.line(overlay, (0, y), (img_w, y), h_color, 1)

            return Image.fromarray(overlay)
        except Exception as e:
            logger.exception(f"[Error] Failed to create annotated image: {e}")
            return original_image

    def _select_best_obbs(
        self,
        image: Image.Image,
        description: str,
        candidate_obbs: List[List[float]],
        sam_metadata: List[Dict[str, Any]],
        masks,
    ) -> Tuple[List[List[float]], List[int]]:
        if not candidate_obbs:
            return [], []

        img_w, img_h = image.size
        annotated_image = self._create_annotated_image(image, masks, sam_metadata)

        # Build text descriptions with normalized coords and geometric features
        obb_descriptions = []
        for meta in sam_metadata:
            norm_obb = self._normalize_obb_to_1000(meta["obb"], img_w, img_h)
            geom = meta["geometric"]
            desc = (
                f"Mask {meta['mask_id']}: "
                f"corners=({norm_obb[0]:.0f},{norm_obb[1]:.0f}), ({norm_obb[2]:.0f},{norm_obb[3]:.0f}), "
                f"({norm_obb[4]:.0f},{norm_obb[5]:.0f}), ({norm_obb[6]:.0f},{norm_obb[7]:.0f}) | "
                f"width_rel={geom['width_rel']:.3f}, height_rel={geom['height_rel']:.3f}, "
                f"area_rel={geom['area_rel']:.4f}, aspect_ratio={geom['aspect_ratio']:.2f}, "
                f"angle={geom['angle']:.1f}°, horizontal={geom['is_horizontal']}, "
                f"compactness={geom['compactness']:.3f}"
            )
            obb_descriptions.append(desc)

        obb_descriptions_text = "\n".join(obb_descriptions)

        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n\n"
            f'TARGET DESCRIPTION: "{description}"\n\n'
            f"CANDIDATE MASKS (shown with colored overlays and numeric IDs):\n"
            f"{obb_descriptions_text}\n\n"
            f"COORDINATE SYSTEM NOTES:\n"
            f"- OBB corners are normalized to 0-1000 range (not pixels)\n"
            f"- (0,0) is at the top-left corner of the image\n"
            f"- width_rel, height_rel, area_rel are normalized relative to image dimensions (0.0 to 1.0)\n"
            f"- aspect_ratio is width/height ratio\n"
            f"- angle is orientation in degrees (0-90°)\n"
            f"- horizontal indicates if object is aligned horizontally (angle near 0° or 90°)\n"
            f"- compactness measures circularity (1.0 = perfect circle, lower = more elongated)\n\n"
            f"GRID OVERLAY:\n"
            f"- The image has a 10x10 grid overlay to help with spatial reference\n"
            f"- Vertical lines are MAGENTA, horizontal lines are CYAN\n\n"
            f"TASK: Identify ALL mask IDs that correspond to the target object(s) described above. "
            f"Consider both the visual appearance in the image and the geometric properties provided.\n"
            f"If no mask corresponds, return an empty string.\n\n"
            f'OUTPUT: Reply with ONLY the mask ID numbers (e.g., "0 3 5 8"). '
            f"Use spaces to separate IDs. No explanation needed."
        )

        logger.info("\n   [Selection] Asking VLM to select best candidate(s)...")
        response = self.vlm.query(annotated_image, prompt_text, max_tokens=20, temperature=0.0)
        logger.info(f"   [Selection] Response: '{response}'")

        matches = re.findall(r"\d+", response)
        selected_indices = []
        selected_obbs = []

        for match in matches:
            try:
                selected_idx = int(match)
                if 0 <= selected_idx < len(candidate_obbs) and selected_idx not in selected_indices:
                    selected_indices.append(selected_idx)
                    selected_obbs.append(candidate_obbs[selected_idx])
            except ValueError:
                continue

        if selected_obbs:
            logger.info(f"    [Selection] Selected Mask IDs: {selected_indices}")
            return selected_obbs, selected_indices

        # Fallback: largest area
        logger.info("    [Selection] Parsing failed or VLM selected none. Falling back to largest mask.")
        if candidate_obbs:
            np = _import_np()
            areas = [meta["area"] for meta in sam_metadata]
            selected_idx = int(np.argmax(areas))
            return [candidate_obbs[selected_idx]], [selected_idx]
        return [], []

    # ------------------------------------------------------------------
    # Stage 3b: VLM direct localization fallback
    # ------------------------------------------------------------------

    def _qwen_direct_localization(self, image: Image.Image, description: str) -> List[List[float]]:
        img_w, img_h = image.size
        prompt_text = (
            f"You are analyzing a remote sensing/aerial image for object localization.\n"
            f'TARGET DESCRIPTION: "{description}"\n'
            f"IMAGE SIZE: {img_w}x{img_h} pixels\n"
            f"NOTE: Coordinate (0,0) is at the top-left corner of the image.\n\n"
            f"TASK: Locate ALL objects described above and provide horizontal bounding boxes for them.\n"
            f"If you find multiple objects, list their coordinates one after another.\n"
            f"If no object is found, return an empty string.\n"
            f"OUTPUT FORMAT: Provide 4 numbers for EACH object: x_min y_min x_max y_max "
            f"(top-left and bottom-right corners)\n"
            f"- x_min: left edge x-coordinate (0 to {img_w})\n"
            f"- y_min: top edge y-coordinate (0 to {img_h})\n"
            f"- x_max: right edge x-coordinate (0 to {img_w})\n"
            f"- y_max: bottom edge y-coordinate (0 to {img_h})\n\n"
            f'Example output for 2 objects: "150 200 300 350 400 450 550 600"\n\n'
            f"Respond with ONLY the numbers separated by spaces, nothing else."
        )

        logger.debug("\n   [Fallback] Segmenter failed. Attempting VLM Direct Localization...")
        response = self.vlm.query(image, prompt_text, max_tokens=100, temperature=0.0)
        logger.info(f"   [Fallback] Response: '{response}'")

        numbers = re.findall(r"-?\d+\.?\d*", response)
        if len(numbers) < 4 or len(numbers) % 4 != 0:
            logger.info(f"    [Fallback] Could not parse valid coordinates ({len(numbers)}).")
            return []

        obbs = []
        for i in range(0, len(numbers), 4):
            try:
                x_min, y_min, x_max, y_max = [float(n) for n in numbers[i : i + 4]]

                # Heuristic: detect normalized 0-1000 coords
                if all(coord <= 1000 for coord in [x_min, y_min, x_max, y_max]) and \
                   any(coord > max(img_w, img_h) for coord in [x_min, y_min, x_max, y_max]):
                    logger.debug("    [Fallback] Detected normalized coordinates, converting to pixels")
                    x_min = (x_min / 1000.0) * img_w
                    y_min = (y_min / 1000.0) * img_h
                    x_max = (x_max / 1000.0) * img_w
                    y_max = (y_max / 1000.0) * img_h

                x_min = max(0, min(img_w, x_min))
                y_min = max(0, min(img_h, y_min))
                x_max = max(0, min(img_w, x_max))
                y_max = max(0, min(img_h, y_max))

                if x_max <= x_min or y_max <= y_min:
                    logger.info(f"    [Fallback] Skipping invalid box: {x_min} {y_min} {x_max} {y_max}")
                    continue

                obb = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                obbs.append(obb)
                logger.debug(f"    [Fallback] Extracted box: ({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")
            except ValueError:
                logger.debug("    [Fallback] Error converting parsed string to float.")
                continue

        if obbs:
            logger.info(f"    [Fallback] Found {len(obbs)} objects via VLM Direct.")
        return obbs

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize(
        self,
        image: Image.Image,
        detections: List[Detection],
        sam_success: bool,
        sam_metadata: List[Dict[str, Any]],
        selected_indices: List[int],
        query: str,
        method: str,
    ) -> None:
        np = _import_np()
        plt = _import_plt()
        from matplotlib.lines import Line2D

        vis_img = np.array(image)
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)

        # Draw final predictions (green)
        for det in detections:
            pred_obb = det.obbox
            obj_id = det.object_id
            pts = np.array(pred_obb).reshape(-1, 2)
            pts = np.vstack((pts, pts[0]))
            plt.plot(pts[:, 0], pts[:, 1], "g-", linewidth=3, label=f"Prediction {obj_id}")
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            plt.text(cx, cy, f"P:{obj_id}", color="white", fontsize=12,
                     bbox=dict(facecolor="green", alpha=0.5))

        # Draw non-selected candidates faintly (yellow dashed)
        if sam_success and method == "segmenter_vlm_select":
            for meta in sam_metadata:
                if meta["mask_id"] not in selected_indices:
                    c_obb = meta["obb"]
                    pts = np.array(c_obb).reshape(-1, 2)
                    pts = np.vstack((pts, pts[0]))
                    plt.plot(pts[:, 0], pts[:, 1], "y--", linewidth=1, alpha=0.7)
                    cx = np.mean(pts[:, 0])
                    cy = np.mean(pts[:, 1])
                    plt.text(cx, cy, str(meta["mask_id"]), color="black", fontsize=8,
                             bbox=dict(facecolor="yellow", alpha=0.4))

        plt.title(f"Result: {query}\nMethod: {method} ({len(detections)} object(s) found)")
        plt.axis("off")
        custom_lines = [Line2D([0], [0], color="g", lw=3),
                        Line2D([0], [0], color="y", linestyle="--", lw=1)]
        plt.legend(custom_lines, ["Final Bounding Box(es)", "Unselected Candidate(s)"], loc="upper right")
        plt.show()
