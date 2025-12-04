"""
Visual Question Answering task implementations.
"""

import re
import torch
from collections import defaultdict
from utils.visualization import annotate_image_with_boxes
from model.tool_calling_step_wise import SatelliteVQAAgent

import logging

logger = logging.getLogger(__name__)


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
9. **Object Existence/Presence**: "Is a X present?", "Are there any X?"

## When to Route to VLM
Route to VLM when the question can be answered through visual understanding alone:
1. **Visual Attributes**: "color", "texture", "appearance"
2. **Object Recognition/Classification**: "What type of...", "Is this urban/rural?"
3. **Scene Understanding**: "weather", "time of day", "context"
4. **Qualitative Descriptions**: "Describe the landscape"
5. **Approximate Comparisons**: "more trees than buildings" (visual estimate)

## Output Format
Respond with ONLY a single word:
- "SAM" - if the question requires SAM3 segmentation
- "VLM" - if the question can be answered by VLM alone
"""

class VQATask:
    """Handles various types of VQA tasks using a Router-based approach."""
    
    def __init__(self, vlm_interface, grounding_task, sam_interface):
        """
        Args:
            vlm_interface: VLMInterface instance
            grounding_task: GroundingTask instance for dynamic grounding
        """
        self.vlm = vlm_interface
        self.grounding = grounding_task
        self.sam_interface = sam_interface
        
        # Initialize the Tool-Calling Agent for SAM-routed tasks
        self.agent = SatelliteVQAAgent(
            vlm_model=vlm_interface.model,
            vlm_processor=vlm_interface.processor,
            sam_interface=self.sam_interface
        )
    
    # --- Entry Points required by RS Pipeline ---
    def answer_question(self, image, query, gsd=1.0, question_type="semantic"):
        return self._answer_integrated(image, query, gsd, question_type)

    # --- Core Routing Logic ---

    def route_question(self, question):
        # [Same as your file]
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswer:"}
        ]
        text = self.vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm.processor(text=[text], padding=True, return_tensors="pt").to(self.vlm.device)
        with torch.no_grad():
            generated_ids = self.vlm.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        output_text = self.vlm.processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return "SAM" if "SAM" in output_text.upper() else "VLM"

    
    def _answer_integrated(self, image, query, gsd, q_type):
        route = self.route_question(query)
        logger.info(f"--- Router Decision: {route} for {q_type} query '{query}' ---")
        
        if route == "SAM":
            return self._answer_via_sam_path(image, query, gsd, q_type)
        else:
            return self._answer_via_vlm_path(image, query, gsd, q_type)

    # --- Solvers ---

    def _answer_via_sam_path(self, image, query, gsd, q_type):
        """SAM Path: Uses Agent with strict format instructions."""
  
        format_instr = ""
        if q_type == "numeric":
            format_instr = "Output ONLY a number."
        elif q_type == "binary":
            format_instr = "Output ONLY 'Yes' or 'No'."
        else:
            format_instr = "Output ONLY a single word/phrase."
            
        augmented_query = f"{query} {format_instr} (GSD: {gsd} m/px)"
        
        response_dict = self.agent.run(image, augmented_query, gsd=gsd)
        
        if "final_answer" in response_dict:
            return response_dict["final_answer"]
        else:
            return "Error"

    
    def _answer_via_vlm_path(self, image, query, gsd, q_type):
        """
        Handles 'VLM' questions: Direct visual understanding with specialized prompts.
        """
        
        visual_input = image
        visual_note = f"The ground sampling distance is {gsd} m/pixel"

        # Select System Prompt based on Question Type
        if q_type == "numeric":
            sys_prompt = (
                "You are a remote sensing assistant. "
                "The user asks a numeric question, to be answered only with a numeric value. "
                "Estimate or count the required value based on the visual image, and all the relevant context from the question. "
                "Provide the number clearly."
            )
        elif q_type == "binary":
            sys_prompt = (
                "You are a remote sensing assistant. "
                "The user asks a binary (Yes/No) question. "
                "Analyze the image and the question context, and answer with ONLY 'Yes' or 'No'."
            )
        else: # semantic
            sys_prompt = (
                "You are a remote sensing assistant. "
                "Answer the question directly and concisely using as few words as possible. "
                "Do not answer with full sentences. "
                "Example: 'Rectangular' instead of 'The field is rectangular'. "
                "Example: 'Blue' instead of 'It is blue'."
            )
        
        user_prompt = f"Question: '{query}'"
        if visual_note:
            user_prompt = f"Context: {visual_note}\n{user_prompt}"

        return self.vlm.query(visual_input, user_prompt, system_prompt=sys_prompt, max_tokens=128)
