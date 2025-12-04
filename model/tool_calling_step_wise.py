import torch
import json
import re
from typing import List, Dict, Any, Union, Tuple
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Sam3Model, Sam3Processor
from qwen_vl_utils import process_vision_info
from utils.geo_calc import GeoCalculator 
from model.model_builder import build_vlm_model, build_sam3_model
from utils.satellite_vqa_tools import select_object_by_rank, calculate_distance_by_indices, calculator_tool

import logging

logger = logging.getLogger(__name__)

SATELLITE_TOOLS = [
    {
        "name": "detect_objects",
        "description": "Step 1: Detects objects in the image using SAM3. MUST be called before any comparison or measurement tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "The object class to detect (e.g., 'building', 'car', 'ship')."
                }
            },
            "required": ["target_class"]
        }
    },
    {
        "name": "get_object_info",
        "description": "Step 2: filters or finds specific objects based on an attribute. Use this to find 'the largest building', 'second smallest car', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "attribute": {
                    "type": "string",
                    "enum": ["area", "length", "confidence", "width", "height"],
                    "description": "The attribute to sort/filter by."
                },
                "rank_index": {
                    "type": "integer",
                    "description": "The rank to select (1-based index). 1 = smallest/first, -1 = largest/last, 2 = second smallest, -2 = second largest."
                }
            },
            "required": ["attribute", "rank_index"]
        }
    },
    {
        "name": "measure_distance",
        "description": "Step 2/3: Measures the distance between two specific objects identified by their ID or rank index from the detection list.",
        "parameters": {
            "type": "object",
            "properties": {
                "index_1": {
                    "type": "integer",
                    "description": "The index of the first object (0-based) from the detected list."
                },
                "index_2": {
                    "type": "integer",
                    "description": "The index of the second object (0-based) from the detected list."
                }
            },
            "required": ["index_1", "index_2"]
        }
    },
    {
        "name": "calculator_tool",
        "description": "Evaluates a mathematical expression to compute totals, ratios, averages, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate (e.g., '(100+200)/5')."
                }
            },
            "required": ["expression"]
        }
    }
]

class SatelliteVQAAgent:
    def __init__(self, vlm_model, vlm_processor, sam_interface=None):
        self.model = vlm_model # vlm_model has been instantiated in rs_pipeline.py
        self.processor = vlm_processor # vlm_processor has been instantiated in rs_pipeline.py
        self.sam = sam_interface 
        self.tools_schema = SATELLITE_TOOLS
        self.tool_map = {
            "detect_objects": self._detect_objects_wrapper,
            "get_object_info": self._get_object_info_wrapper,
            "measure_distance": self._measure_distance_wrapper,
            "calculator_tool": calculator_tool,
        }
        self.image = None
        self.current_gsd = 1.0

        self.sam_state = {
            "objects": [],    
            "masks": None,   
            "count": 0,
            "image_size": (0,0)
        }
        logger.info("Instantiating SatelliteVQAAgent")

    def _reset_state(self):
        """Clears the SAM state for a new query."""
        self.sam_state = {
            "objects": [],
            "masks": None,
            "count": 0,
            "image_size": (0,0)
        }

    def _detect_objects_wrapper(self, target_class):
        """
        Runs SAM and populates self.sam_state.
        """
        logger.info(f"Running SAM for class: {target_class}")
        result = self.sam.segment_image(self.image, target_class, gsd=self.current_gsd)
        
        if result and result.get("metadata"):
            # Update State
            self.sam_state["objects"] = result["metadata"]
            self.sam_state["masks"] = result["masks"]
            self.sam_state["count"] = result["count"]
            self.sam_state["image_size"] = result["image_size"]
            
            # Return a summary to Qwen (not the raw data)
            return f"Success. Detected {result['count']} instances of '{target_class}'. These are now stored in memory with indices 0 to {result['count']-1}."
        else:
            self._reset_state()
            return f"No instances of '{target_class}' were detected."
            
    def _get_object_info_wrapper(self, attribute, rank_index):
        """
        Uses self.sam_state implicitly.
        Qwen asks for: attribute='area', rank_index=-2 (2nd largest).
        """
        objects = self.sam_state["objects"]
        if not objects:
            return "Error: No objects detected yet. Please call 'detect_objects' first."

        # Map 'area' to specific key if needed, or use directly
        attr_map = {
            "area": "area_m2",
            "length": "height_m", # Assuming height is length in OBB
            "width": "width_m",
            "confidence": "confidence"
        }
        key = attr_map.get(attribute, attribute)

        # Call helper logic (see satellite_vqa_tools.py)
        selected_obj, info_str = select_object_by_rank(objects, key, rank_index)
        
        return info_str

    def _measure_distance_wrapper(self, index_1, index_2):
        """
        Uses self.sam_state implicitly.
        Qwen asks for: index_1=0, index_2=3.
        """
        objects = self.sam_state["objects"]
        if not objects:
            return "Error: No objects detected yet."
            
        dist_val = calculate_distance_by_indices(objects, index_1, index_2)
        dist_meters = dist_pixels * self.current_gsd
        return f"Distance: {dist_meters:.2f} meters."

    def _format_system_prompt(self):
        tools_json = json.dumps(self.tools_schema, indent=2)
        
        prompt = f"""You are a Satellite Imagery Analysis Agent.
        
        You have access to a set of TOOLS.
        state: The system maintains an internal memory of detected objects. 
        
        TOOLS:
        {tools_json}

        INSTRUCTIONS:
        1. Always detect objects first if the question implies specific items (cars, buildings).
        2. Once detected, refer to objects by their implied properties (rank, size) using 'get_object_info'.
        3. Do NOT try to estimate coordinates or areas yourself. Use the tools.
        4. If you need a tool, output a JSON object with "tool" and "arguments".
        
        Example Flow:
        User: "How big is the second largest building?"
        Step 1: {{ "tool": "detect_objects", "arguments": {{ "target_class": "building" }} }}
        Obs: Success. Detected 15 buildings.
        Step 2: {{ "tool": "get_object_info", "arguments": {{ "attribute": "area", "rank_index": -2 }} }}
        """
        return prompt

    def run(self, image: Image.Image, user_query: str, gsd: float = 1.0, metadata: Dict = {}, max_steps: int = 5):
        self.image = image
        self.current_gsd = gsd
        self._reset_state() # Clear previous query state
        
        system_prompt_text = self._format_system_prompt()
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_query}]}
        ]

        logger.info(f"--- Starting Agent Loop (Max Steps: {max_steps}) ---")

        for step in range(max_steps):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            logger.info(f"\n[Step {step+1} Model Output]: {output_text}")

            tool_result = self._parse_and_execute_tool(output_text)
            
            if tool_result:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                
                # Format observation
                observation_text = f"Observation from tool '{tool_result['tool']}': {tool_result['result']}"
                messages.append({"role": "user", "content": [{"type": "text", "text": observation_text}]})
                
                logger.info(f"[System]: {observation_text}")
            else:
                return {
                    "final_answer": output_text,
                    "steps_taken": step + 1
                }

        return {"error": "Max steps reached without final answer."}
        

    def _parse_and_execute_tool(self, text: str):
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                tool_name = data.get("tool")
                arguments = data.get("arguments")
                
                if tool_name in self.tool_map:
                    func = self.tool_map[tool_name]
                    result = func(**arguments)
                    return {"tool": tool_name, "result": result}
        except Exception as e:
            logger.info(f"Failed to parse or execute tool: {e}")
            return None
        return None

'''
if __name__ == "__main__":

    vlm_model, vlm_processor = build_vlm_model()
    sam_model, sam_processor = build_sam3_model()
    sam_interface = SAM3Interface(sam_model, sam_processor)
    
    agent = SatelliteVQAAgent(vlm_model, vlm_processor, sam_interface)
    
    # 3. Mock Data
    mock_image = Image.new('RGB', (500, 500), color='white')
    mock_metadata = {
        "objects": [
            {"id": "car_1", "class": "car", "area": 12.5, "obb_coords": [10, 20, 10, 20]},
            {"id": "car_2", "class": "car", "area": 14.0, "obb_coords": [30, 40, 30, 40]},
            {"id": "bldg_1", "class": "building", "area": 500.0, "obb_coords": [100, 200, 100, 200]}
        ]
    }
    
    print("Agent setup complete")

'''





