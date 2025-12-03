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

try:
    from satellite_vqa_tools import comparison_tool, distance_tool, calculator_tool
except ImportError:
    print("Warning: satellite_vqa_tools.py not found. Tool execution will fail.")



SATELLITE_TOOLS = [
    {
        "name": "comparison_tool",
        "description": "Sorts a list of object ID and value pairs. Useful for finding largest/smallest objects or ranking objects by a specific attribute (area, height, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "data_pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "Tuple of [object_id, value]"
                    },
                    "description": "List of (id, value) tuples to sort."
                },
                "descending": {
                    "type": "boolean",
                    "description": "True for largest-first, False for smallest-first. Default False."
                }
            },
            "required": ["data_pairs"]
        }
    },
    {
        "name": "distance_tool",
        "description": "Calculates Euclidean distances between two groups of objects given their bounding boxes.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_a": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "Tuple of [id, x1, x2, y1, y2]"
                    },
                    "description": "First list of objects with bounding box coordinates."
                },
                "group_b": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "Tuple of [id, x1, x2, y1, y2]"
                    },
                    "description": "Second list of objects to compare against."
                }
            },
            "required": ["group_a", "group_b"]
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
    },
    {
        "name": "SAM_tool",
        "description": "Uses SAM3 to detect objects in the image. Takes in a noun phrase as an input in the target_class argument and returns mask id, oriented bounding box, confidence, and area of the object for each instance of the object in the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "The target class to use as a text prompt for SAM3."
                }
            },
            "required": ["target_class"]
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
            "comparison_tool": comparison_tool,
            "distance_tool": distance_tool,
            "calculator_tool": calculator_tool,
            "SAM_tool": self._sam_tool_wrapper
        }
        self.image = None


    def _sam_tool_wrapper(self, target_class):
        """Wrapper so the tool API only needs target_class."""
        return self.sam.segment_image(self.image, target_class)

    def _format_system_prompt(self, metadata: Dict):
        """
        Creates the system prompt injecting the tool definitions and image metadata.
        """
        tools_json = json.dumps(self.tools_schema, indent=2)
        metadata_json = json.dumps(metadata, indent=2)
        
        prompt = f"""You are a Satellite Imagery Analysis Agent.
        
        You have access to the following TOOLS to answer user questions:
        {tools_json}

        You are provided with METADATA about objects detected in the image:
        {metadata_json}

        INSTRUCTIONS:
        1. Analyze the user query.
        2. Check the METADATA to extract relevant IDs and numerical values (coordinates, areas, etc.).
        3. Decide if you need to use a tool.
        4. If you need a tool, output a JSON object with the keys "tool" and "arguments".
        Example: {{ "tool": "calculator_tool", "arguments": {{ "expression": "10 + 20" }} }}
        5. If no tool is needed, answer directly.
        """
        
        return prompt

    def run(self, image: Image.Image, user_query: str, metadata: Dict, max_steps: int = 5):
        """
        Executes the agent loop with Multi-Step capability (ReAct Loop).
        """
        self.image = image
        system_prompt_text = self._format_system_prompt(metadata)
        
        # 1. Initialize History
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_query}
                ]
            }
        ]

        print(f"--- Starting Agent Loop (Max Steps: {max_steps}) ---")

        for step in range(max_steps):
            # 2. Prepare Inputs (History -> Tensors)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 3. Generate Model Output
            # print(f"Step {step + 1}: Generating thought...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(f"\n[Step {step+1} Model Output]: {output_text}")

            # 4. Check for Tool Call
            tool_result = self._parse_and_execute_tool(output_text)
            
            if tool_result:
                
                # A. Append the model's thought/call as 'assistant'
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}]
                })
                
                # B. Append the tool execution result as 'user' (Observation)
                observation_text = f"Observation from tool '{tool_result['tool']}': {tool_result['result']}"
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": observation_text}]
                })
                
                print(f"[System]: {observation_text}")
                # Loop continues to next iteration (Step 2, 3...)
                
            else:
                # No tool call found -> This is the Final Answer
                print("\n[Final Answer Reached]")
                return {
                    "final_answer": output_text,
                    "steps_taken": step + 1
                }

        return {"error": "Max steps reached without final answer."}

    def _parse_and_execute_tool(self, text: str):
        """
        Parses JSON tool calls from text and executes the corresponding Python function.
        """
        try:
            # Look for JSON block
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                tool_name = data.get("tool")
                arguments = data.get("arguments")
                
                if tool_name in self.tool_map:
                    func = self.tool_map[tool_name]
                    # Unpack arguments into function
                    result = func(**arguments)
                    return {"tool": tool_name, "result": result}
                else:
                    return None
        except Exception as e:
            print(f"Failed to parse or execute tool: {e}")
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