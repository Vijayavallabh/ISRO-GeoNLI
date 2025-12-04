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
        "description": "Step 1: Detects objects. Returns count and stores attributes (area, shape, orientation, width, height) in memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "The object class (e.g., 'building', 'car', 'ship')."
                }
            },
            "required": ["target_class"]
        }
    },
    {
        "name": "get_object_info",
        "description": "Step 2: Finds a specific object based on a sorting criteria and returns a specific attribute.",
        "parameters": {
            "type": "object",
            "properties": {
                "sort_attribute": {
                    "type": "string",
                    "enum": ["area", "length", "confidence", "width", "height"],
                    "description": "The attribute to use for ranking/sorting the objects."
                },
                "rank_index": {
                    "type": "integer",
                    "description": "1 = smallest/first, -1 = largest/last, 2 = second smallest."
                },
                "return_attribute": {
                    "type": "string",
                    "enum": ["area", "length", "width", "height", "shape", "orientation", "confidence"],
                    "description": "The actual attribute value to return. If omitted, returns the sort_attribute value."
                }
            },
            "required": ["sort_attribute", "rank_index"]
        }
    },
    {
        "name": "measure_distance",
        "description": "Step 2/3: Measures distance between two objects by their ID/Index.",
        "parameters": {
            "type": "object",
            "properties": {
                "index_1": {
                    "type": "integer",
                    "description": "Index of first object."
                },
                "index_2": {
                    "type": "integer",
                    "description": "Index of second object."
                }
            },
            "required": ["index_1", "index_2"]
        }
    },
    {
        "name": "calculator_tool",
        "description": "Evaluates math expressions (e.g. for averages, sums).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '(100+200)/5')."
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
        logger.info(f"Running SAM for class: {target_class}")
        result = self.sam.segment_image(self.image, target_class, gsd=self.current_gsd)
        
        if result and result.get("metadata"):
            self.sam_state["objects"] = result["metadata"]
            self.sam_state["masks"] = result["masks"]
            self.sam_state["count"] = result["count"]
            self.sam_state["image_size"] = result["image_size"]
            
            # Explicitly tell Qwen that attributes are ready
            return (f"Success. Detected {result['count']} '{target_class}'(s). "
                    f"Indices: 0 to {result['count']-1}. "
                    "Attributes available in memory: area, shape, orientation, width, height.")
        else:
            self._reset_state()
            return f"No instances of '{target_class}' were detected."
            
    def _get_object_info_wrapper(self, sort_attribute, rank_index, return_attribute=None):
        objects = self.sam_state["objects"]
        if not objects:
            return "Error: No objects detected yet. Please call 'detect_objects' first."

        # Map generic terms to internal keys
        attr_map = {
            "area": "area_m2",
            "length": "height_m", 
            "width": "width_m",
            "height": "height_m",
            "confidence": "confidence",
            "shape": "shape",
            "orientation": "orientation_deg"
        }
        
        sort_key = attr_map.get(sort_attribute, sort_attribute)
        return_key = attr_map.get(return_attribute, return_attribute) if return_attribute else sort_key

        selected_obj, info_str = select_object_by_rank(objects, sort_key, rank_index, return_key)
        return info_str

    def _measure_distance_wrapper(self, index_1, index_2):
        objects = self.sam_state["objects"]
        if not objects:
            return "Error: No objects detected yet."
            
        dist_pixels = calculate_distance_by_indices(objects, index_1, index_2)
        if dist_pixels is None:
             return "Error: Invalid indices provided."

        dist_meters = dist_pixels * self.current_gsd
        return f"{dist_meters:.2f}"

    def _format_system_prompt(self):
        tools_json = json.dumps(self.tools_schema, indent=2)
        
        prompt = f"""You are a Satellite Analysis Agent. 
        You have access to tools to analyze images.
        
        TOOLS:
        {tools_json}

        CRITICAL OUTPUT RULES:
        1. If the user asks for a specific attribute (e.g. "What is the shape?"), use 'get_object_info' to retrieve it.
        2. Do NOT guess attributes. Use the tools.
        3. FINAL ANSWER FORMAT: 
           - Numeric questions: Return ONLY the number (e.g. "450.5").
           - Binary questions: Return ONLY "Yes" or "No".
           - Semantic questions: Return ONLY the single word or short phrase (e.g., "Rectangle", "North").
           - Do not write full sentences in the final answer.

        Example Flow:
        User: "What is the shape of the largest building?"
        Step 1: {{ "tool": "detect_objects", "arguments": {{ "target_class": "building" }} }}
        Obs: Success. Detected 5 buildings. Attributes available.
        Step 2: {{ "tool": "get_object_info", "arguments": {{ "sort_attribute": "area", "rank_index": -1, "return_attribute": "shape" }} }}
        Obs: rectangle
        Final Answer: rectangle
        """
        return prompt

    def run(self, image: Image.Image, user_query: str, gsd: float = 1.0, max_steps: int = 5):
        self.image = image
        self.current_gsd = gsd
        self._reset_state()
        
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
                observation_text = f"Observation from tool '{tool_result['tool']}': {tool_result['result']}"
                messages.append({"role": "user", "content": [{"type": "text", "text": observation_text}]})
                logger.info(f"[System]: {observation_text}")
            else:
                # Final clean up to ensure no punctuation or filler remains if Qwen forgets
                clean_answer = output_text.strip().rstrip('.')
                return {
                    "final_answer": clean_answer,
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






