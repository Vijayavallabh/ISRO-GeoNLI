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
from utils.satellite_vqa_tools import select_object_by_rank, calculate_distance_by_indices, calculator_tool, filter_objects_by_region

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
        "name": "filter_objects",
        "description": "Step 2 (Optional): Filters the currently detected objects by spatial region (e.g., 'top left', 'center').",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "The region to filter by: 'top', 'bottom', 'left', 'right', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'."
                }
            },
            "required": ["region"]
        }
    },
    {
        "name": "get_object_info",
        "description": "Step 3: Finds a specific object from the (filtered) list based on sorting/ranking.",
        "parameters": {
            "type": "object",
            "properties": {
                "sort_attribute": {
                    "type": "string",
                    "enum": ["area", "length", "confidence", "width", "height"],
                    "description": "The attribute to use for sorting."
                },
                "rank_index": {
                    "type": "integer",
                    "description": "1 = smallest/first, -1 = largest/last, 2 = second smallest."
                },
                "return_attribute": {
                    "type": "string",
                    "enum": ["area", "length", "width", "height", "shape", "orientation", "confidence", "mask_id"],
                    "description": "The attribute to return. Use 'mask_id' to get the ID."
                }
            },
            "required": ["sort_attribute", "rank_index"]
        }
    },
    {
        "name": "select_final_object",
        "description": "Final Step for Grounding: Selects the specific object ID that answers the user's description.",
        "parameters": {
            "type": "object",
            "properties": {
                "mask_id": {
                    "type": "integer",
                    "description": "The mask_id of the object to select."
                }
            },
            "required": ["mask_id"]
        }
    },
    {
        "name": "measure_distance",
        "description": "Measures distance between two objects by their ID/Index.",
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
        "description": "Evaluates math expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression."
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
            "filter_objects": self._filter_objects_wrapper,
            "get_object_info": self._get_object_info_wrapper,
            "select_final_object": self._select_final_object_wrapper,
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
        self.final_selected_id = None
        logger.info("Instantiating SatelliteVQAAgent")

    def _reset_state(self):
        self.sam_state = {"objects": [], "all_objects": [], "masks": None, "image_size": (0,0)}
        self.final_selected_id = None

    def _detect_objects_wrapper(self, target_class):
        logger.info(f"Running SAM for class: {target_class}")
        result = self.sam.segment_image(self.image, target_class, gsd=self.current_gsd)
        
        if result and result.get("metadata"):
            self.sam_state["objects"] = result["metadata"]
            self.sam_state["all_objects"] = result["metadata"] # Backup
            self.sam_state["masks"] = result["masks"]
            self.sam_state["image_size"] = result["image_size"]
            
            return (f"Success. Detected {result['count']} '{target_class}'(s). "
                    f"Indices: 0 to {result['count']-1}. "
                    "Attributes available: area, shape, orientation, width, height.")
        else:
            self._reset_state()
            return f"No instances of '{target_class}' were detected."

    def _filter_objects_wrapper(self, region):
        """Filters the *current* list of objects in sam_state."""
        current_objs = self.sam_state["objects"]
        if not current_objs:
            return "Error: No objects to filter. Detect first."
            
        filtered = filter_objects_by_region(current_objs, region, self.sam_state["image_size"])
        
        # Update the working list
        self.sam_state["objects"] = filtered
        
        return (f"Filtered by region '{region}'. Remaining objects: {len(filtered)}. "
                f"You can now use 'get_object_info' on this filtered list.")
        
            
    def _get_object_info_wrapper(self, sort_attribute, rank_index, return_attribute=None):
        objects = self.sam_state["objects"]
        if not objects:
            return "Error: No objects detected/remaining."

        attr_map = {
            "area": "area_m2",
            "length": "height_m", 
            "width": "width_m",
            "height": "height_m",
            "confidence": "confidence",
            "shape": "shape",
            "orientation": "orientation_deg",
            "mask_id": "mask_id"
        }
        
        sort_key = attr_map.get(sort_attribute, sort_attribute)
        return_key = attr_map.get(return_attribute, return_attribute) if return_attribute else sort_key

        selected_obj, info_str = select_object_by_rank(objects, sort_key, rank_index, return_key)
        
        if selected_obj:
            # Add mask_id to the info string if not asked for, to help the agent know what ID it is
            if "mask_id" not in info_str and "ID" not in info_str:
                info_str += f" (Mask ID: {selected_obj.get('mask_id')})"
                
        return info_str

    def _select_final_object_wrapper(self, mask_id):
        self.final_selected_id = int(mask_id)
        return f"Object {mask_id} selected as final answer."
        

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
        
        TOOLS:
        {tools_json}

        INSTRUCTIONS:
        1. For GROUNDING (locating specific objects):
           - Step 1: 'detect_objects'
           - Step 2: 'filter_objects' (if location is mentioned like "top left")
           - Step 3: 'get_object_info' (to find largest/smallest etc.)
           - Step 4: 'select_final_object' (CRITICAL: Call this with the ID you found)
        
        2. For VQA (answering questions):
           - Use tools to gather info, then answer in the Final Answer.
           - Numeric: ONLY number.
           - Binary: ONLY Yes/No.
           - Semantic: ONLY single word/phrase.

        Example Grounding:
        User: "Find the largest car in the top left."
        Step 1: detect_objects("car")
        Step 2: filter_objects("top-left")
        Step 3: get_object_info("area", -1, "mask_id") -> "Mask ID: 5"
        Step 4: select_final_object(5)
        Final Answer: Found object 5.
        """
        return prompt

    def run(self, image: Image.Image, user_query: str, gsd: float = 1.0, max_steps: int = 6):
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
                
                # If the agent selected a final object, we can stop early if we want, or let it generate the text confirmation.
                if tool_result["tool"] == "select_final_object":
                    logger.info("Agent selected final object. Stopping.")
                    return {
                        "final_answer": output_text,
                        "selected_object_id": self.final_selected_id,
                        "sam_state_objects": self.sam_state["all_objects"] # Return full list to retrieve bbox
                    }
                    
            else:
                return {
                    "final_answer": output_text.strip(),
                    "selected_object_id": self.final_selected_id, # Might be None for VQA
                    "sam_state_objects": self.sam_state["all_objects"]
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







