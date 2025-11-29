"""
Visual Question Answering task implementations.
"""

import re
from collections import defaultdict
from utils.visualization import annotate_image_with_boxes


class VQATask:
    """Handles various types of VQA tasks."""
    
    def __init__(self, vlm_interface, grounding_task):
        """
        Args:
            vlm_interface: VLMInterface instance
            grounding_task: GroundingTask instance for dynamic grounding
        """
        self.vlm = vlm_interface
        self.grounding = grounding_task
    
    def answer_numeric_question(self, image, query, detections, gsd=1.0):
        """
        Answer numeric questions (counting, area, distance).
        
        Args:
            image: PIL Image
            query: Question string
            detections: List of existing detections
            gsd: Ground Sample Distance
            
        Returns:
            Numeric answer as string
        """
        print(f"--- Task: Numeric VQA ('{query}') ---")
        
        # Ensure we have necessary detections
        detections = self._ensure_detections(image, query, detections, gsd)
        
        # Prepare context
        context_str = self._format_detection_context(detections)
        
        # Define strict system behavior
        sys_prompt = (
            "You are a helpful AI assistant acting as a calculator. "
            "You will be provided with a list of detected objects and their metadata "
            "(Area, Coordinates). "
            "Your goal is to answer the user's numeric question using ONLY this metadata. "
            "Perform the calculation internally and output ONLY the final number. "
            "Do not output units, equations, or sentences."
        )
        
        user_prompt = (
            f"Metadata Context:\n{context_str}\n\n"
            f"Question: '{query}'\n"
            "Answer:"
        )
        
        answer = self.vlm.query(image, user_prompt, 
                               system_prompt=sys_prompt, max_tokens=32)
        
        # Clean up any lingering text
        cleaned_answer = re.sub(r"[^\d\.]", "", answer)
        return cleaned_answer
    
    def answer_binary_question(self, image, query, detections, gsd=1.0):
        """
        Answer binary yes/no questions.
        
        Args:
            image: PIL Image
            query: Question string
            detections: List of existing detections
            gsd: Ground Sample Distance
            
        Returns:
            'Yes' or 'No'
        """
        return self._answer_general_vqa(
            image, query, detections, gsd, 
            question_type="binary"
        )
    
    def answer_semantic_question(self, image, query, detections, gsd=1.0):
        """
        Answer semantic questions (color, material, activity).
        
        Args:
            image: PIL Image
            query: Question string
            detections: List of existing detections
            gsd: Ground Sample Distance
            
        Returns:
            Short text answer
        """
        return self._answer_general_vqa(
            image, query, detections, gsd,
            question_type="semantic"
        )
    
    def _answer_general_vqa(self, image, query, detections, gsd, question_type):
        """Internal method for binary/semantic VQA."""
        print(f"--- Task: {question_type.capitalize()} VQA ('{query}') ---")
        
        # Ensure detections
        detections = self._ensure_detections(image, query, detections, gsd)
        
        # Prepare visuals
        target_keywords = self.grounding.extract_target_classes(image, query)
        relevant_obbs = [d['obb'] for d in detections 
                        if d['label'] in target_keywords]
        
        if relevant_obbs:
            visual_input, _ = annotate_image_with_boxes(image, relevant_obbs)
            visual_note = ("The image has been annotated with RED BOXES and IDs "
                          "to help you locate the objects.")
        else:
            all_obbs = [d['obb'] for d in detections]
            if all_obbs:
                visual_input, _ = annotate_image_with_boxes(image, all_obbs)
                visual_note = "The image is annotated with all detected objects."
            else:
                visual_input = image
                visual_note = "No specific objects were detected in the metadata."
        
        # Define system behavior
        if question_type == "binary":
            sys_prompt = ("You are a strict answering machine. "
                         "Answer the question with 'Yes' or 'No' ONLY.")
        else:
            sys_prompt = ("You are a concise assistant. "
                         "Answer the question with a single word or short phrase.")
        
        context_str = self._format_detection_context(detections)
        
        user_prompt = (
            f"Metadata Context:\n{context_str}\n\n"
            f"Visual Context: {visual_note}\n"
            f"Question: '{query}'"
        )
        
        return self.vlm.query(visual_input, user_prompt, 
                             system_prompt=sys_prompt, max_tokens=32)
    
    def _ensure_detections(self, image, query, current_detections, gsd):
        """Dynamically ground missing objects if needed."""
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
