"""
Visual Question Answering task implementations.
"""

import re
import torch
from collections import defaultdict
from utils.visualization import annotate_image_with_boxes


# --- ROUTER CONFIGURATION ---
ROUTER_SYSTEM_PROMPT = """You are an intelligent routing system for remote sensing visual question answering tasks. Your job is to analyze questions and determine whether they require SAM3 (Segment Anything Model 3) segmentation capabilities or can be answered directly by a Vision Language Model (VLM).

## SAM3 Capabilities
SAM3 is a foundation model for promptable segmentation that can:
- Detect and segment ALL instances of objects specified by text prompts (e.g., "buildings", "trees", "vehicles")
- Generate precise pixel-level segmentation masks for multiple object instances
- Provide bounding boxes and confidence scores for detected objects
- Return masks that enable precise geometric calculations (area, perimeter, length, orientation)

## When to Route to SAM3
Route to SAM3 when the question requires:
1. **Counting/Quantification**: "how many", "number of", "count", "amount of"
2. **Area/Coverage Calculations**: "area covered", "percentage of coverage", "spatial extent"
3. **Length/Distance Measurements**: "length", "width", "perimeter", "distance"
4. **Orientation/Angle Analysis**: "direction", "orientation", "angle"
5. **Density/Concentration**: "density", "distribution"
6. **Ratio/Proportion Comparisons**: "ratio of", "more X than Y"
7. **Spatial Relationships Requiring Segmentation**: "adjacent to", "overlap", "precise arrangement"
8. **Size/Dimension Analysis**: "size of", "how wide"

## When to Route to VLM
Route to VLM when the question can be answered through visual understanding alone:
1. **Object Existence/Presence**: "Is a X present?", "Are there any X?"
2. **Visual Attributes**: "color", "texture", "appearance"
3. **Object Recognition/Classification**: "What type of...", "Is this urban/rural?"
4. **Scene Understanding**: "weather", "time of day", "context"
5. **Qualitative Descriptions**: "Describe the landscape"
6. **Approximate Comparisons**: "more trees than buildings" (visual estimate)

## Output Format
Respond with ONLY a single word:
- "SAM" - if the question requires SAM3 segmentation
- "VLM" - if the question can be answered by VLM alone
"""

class VQATask:
    """Handles various types of VQA tasks using a Router-based approach."""
    
    def __init__(self, vlm_interface, grounding_task):
        """
        Args:
            vlm_interface: VLMInterface instance
            grounding_task: GroundingTask instance for dynamic grounding
        """
        self.vlm = vlm_interface
        self.grounding = grounding_task
    
    def route_question(self, question):
        """
        Determines if a question needs SAM (metadata/grounding) or VLM (visual only).
        """
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswer:"}
        ]
        
        # Prepare inputs using the VLM's processor
        text = self.vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.vlm.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        
        # Generate routing decision
        with torch.no_grad():
            generated_ids = self.vlm.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1, # Low temp for consistency
                do_sample=False,
            )
            
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.vlm.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        route = output_text.strip().upper()
        return "SAM" if "SAM" in route else "VLM"

    def answer_question(self, image, query, detections=None, gsd=1.0):
        """
        Unified method that routes the question to the appropriate solver.
        """
        # 1. Decide on the strategy
        route = self.route_question(query)
        print(f"--- Router Decision: {route} for query '{query}' ---")
        
        # 2. Execute the strategy
        if route == "SAM":
            return self._answer_via_sam_path(image, query, detections, gsd)
        else:
            return self._answer_via_vlm_path(image, query, detections)

    # --- Solvers ---

    def _answer_via_sam_path(self, image, query, detections, gsd):
        """
        Handles 'SAM' questions: Requires precise object grounding and metadata.
        Focuses on counts, areas, and geometric properties.
        """
        # Ensure we have necessary detections (Critical for SAM path)
        detections = self._ensure_detections(image, query, detections, gsd)
        
        # Prepare Metadata Context
        context_str = self._format_detection_context(detections)
        
        # Visual Annotation (Optional but helpful context)
        all_obbs = [d['obb'] for d in detections]
        if all_obbs:
            visual_input, _ = annotate_image_with_boxes(image, all_obbs)
            visual_note = "The image is annotated with detected objects."
        else:
            visual_input = image
            visual_note = "No specific objects were detected."

        # Strict System Prompt for Calculation/Logic
        sys_prompt = (
            "You are an intelligent remote sensing analyst acting as a calculator. "
            "You have access to detailed object metadata (counts, areas, coordinates) in the context. "
            "Use this metadata to answer the question accurately. "
            "If the question asks for a count, area, or calculation, derive it STRICTLY from the metadata. "
            "Do not output units, equations, or filler text unless asked. Output the final answer concisely."
        )
        
        user_prompt = (
            f"Metadata Context:\n{context_str}\n\n"
            f"Visual Context: {visual_note}\n"
            f"Question: '{query}'"
        )
        
        return self.vlm.query(visual_input, user_prompt, system_prompt=sys_prompt, max_tokens=128)

    def _answer_via_vlm_path(self, image, query, detections):
        """
        Handles 'VLM' questions: Visual understanding, existence, description.
        Does NOT rely on precise metadata or counting logic.
        """
        # If detections exist, we can use them for visual context, but we won't force new ones.
        if detections:
            # Filter relevant ones if possible, or just show all
            all_obbs = [d['obb'] for d in detections]
            visual_input, _ = annotate_image_with_boxes(image, all_obbs)
            visual_note = "The image contains some annotated objects, but rely mainly on your visual perception."
        else:
            visual_input = image
            visual_note = ""

        # General concise system prompt
        sys_prompt = (
            "You are a helpful and concise remote sensing assistant. "
            "Answer the question based on the visual content of the image. "
            "Keep the answer short and direct."
        )
        
        user_prompt = f"Question: '{query}'"
        if visual_note:
            user_prompt = f"Context: {visual_note}\n{user_prompt}"

        return self.vlm.query(visual_input, user_prompt, system_prompt=sys_prompt, max_tokens=128)

    # --- Legacy Compatibility Methods (Redirect to Router) ---
    
    def answer_numeric_question(self, image, query, detections, gsd=1.0):
        """Legacy entry point: Redirects to unified router logic."""
        return self.answer_question(image, query, detections, gsd)
    
    def answer_binary_question(self, image, query, detections, gsd=1.0):
        """Legacy entry point: Redirects to unified router logic."""
        return self.answer_question(image, query, detections, gsd)
    
    def answer_semantic_question(self, image, query, detections, gsd=1.0):
        """Legacy entry point: Redirects to unified router logic."""
        return self.answer_question(image, query, detections, gsd)
    
    # --- Helpers ---

    def _ensure_detections(self, image, query, current_detections, gsd):
        """Dynamically ground missing objects if needed."""
        if current_detections is None:
            current_detections = []
            
        needed_targets = self.grounding.extract_target_classes(image, query)
        existing_labels = {d['label'] for d in current_detections}
        missing_targets = [t for t in needed_targets if t not in existing_labels]
        
        if not missing_targets:
            return current_detections
        
        print(f"   [Dynamic Grounding] VQA needs {missing_targets}. Scanning...")
        new_detections = self.grounding.ground_objects(
            image, query, gsd=gsd, 
            forced_targets=missing_targets,
            show_visualization=False
        )
        
        return current_detections + new_detections
    
    def _format_detection_context(self, detections):
        """Format detection metadata as text context."""
        if not detections:
            return "No objects were detected in the grounding phase."
        
        grouped = defaultdict(list)
        for det in detections:
            grouped[det['label']].append(det)
        
        lines = ["Grounding Phase Results:"]
        
        summary_parts = []
        for label, dets in grouped.items():
            summary_parts.append(f"{len(dets)} {label}(s)")
        lines.append("Summary: Found " + ", ".join(summary_parts) + ".")
        lines.append("-" * 30)
        
        for label, dets in grouped.items():
            lines.append(f"Category: '{label}'")
            for det in dets:
                cx, cy = det['center_point']
                lines.append(
                    f"  - ID {det['id']}: "
                    f"Area={det['area_m2']:.2f}m2, "
                    f"Center=({int(cx)}, {int(cy)}), "
                    f"Angle={det['obb'][2]:.1f}"
                )
        
        return "\n".join(lines)
