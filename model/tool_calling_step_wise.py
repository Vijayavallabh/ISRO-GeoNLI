import torch
import json
import re
from typing import List, Dict, Any, Union
from PIL import Image
from qwen_vl_utils import process_vision_info
from utils.satellite_vqa_tools import (
    select_object_by_rank, 
    calculate_distance_by_indices, 
    calculator_tool,
    get_available_attributes
)

import logging

logger = logging.getLogger(__name__)

SATELLITE_TOOLS = [
    {
        "name": "detect_objects",
        "description": (
            "STEP 1: Detects all instances of a target object class in the image. "
            "Returns count and stores metadata (area, width, height, shape, orientation, confidence) "
            "for each detected object. Must be called before other tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "Object class to detect (e.g., 'building', 'vehicle', 'tree', 'ship')"
                }
            },
            "required": ["target_class"]
        }
    },
    {
        "name": "get_object_info",
        "description": (
            "STEP 2: Retrieves a specific attribute from an object selected by ranking. "
            "Objects are sorted by sort_attribute, then the object at rank_index is selected, "
            "and return_attribute is retrieved from that object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sort_attribute": {
                    "type": "string",
                    "enum": ["area", "width", "height", "orientation", "confidence"],
                    "description": "Attribute to sort objects by before selecting"
                },
                "rank_index": {
                    "type": "integer",
                    "description": (
                        "Which ranked object to select: "
                        "1 = smallest/first, -1 = largest/last, "
                        "2 = second smallest, -2 = second largest"
                    )
                },
                "return_attribute": {
                    "type": "string",
                    "enum": ["area", "width", "height", "shape", "orientation", "confidence"],
                    "description": (
                        "The attribute to return from the selected object. "
                        "If omitted, returns the sort_attribute value."
                    )
                }
            },
            "required": ["sort_attribute", "rank_index"]
        }
    },
    {
        "name": "measure_distance",
        "description": (
            "STEP 2/3: Measures the distance in meters between centroids of two objects. "
            "Objects are identified by their 0-based index (0 to count-1)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index_1": {
                    "type": "integer",
                    "description": "Index of first object (0-based)"
                },
                "index_2": {
                    "type": "integer",
                    "description": "Index of second object (0-based)"
                }
            },
            "required": ["index_1", "index_2"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "STEP 3: Evaluates mathematical expressions. "
            "Useful for computing averages, sums, ratios, etc. "
            "Supports: +, -, *, /, **, sqrt(), abs(), round(), min(), max()"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '(100 + 200) / 2')"
                }
            },
            "required": ["expression"]
        }
    }
]


class SatelliteVQAAgent:
    def __init__(self, vlm_model, vlm_processor, sam_interface):
        self.model = vlm_model
        self.processor = vlm_processor
        self.sam = sam_interface
        self.tools_schema = SATELLITE_TOOLS
        
        # Map tool names to wrapper functions
        self.tool_map = {
            "detect_objects": self._detect_objects_wrapper,
            "get_object_info": self._get_object_info_wrapper,
            "measure_distance": self._measure_distance_wrapper,
            "calculate": self._calculate_wrapper,
        }
        
        # State variables
        self.image = None
        self.current_gsd = 1.0
        self.sam_state = {
            "objects": [],      # List of object metadata dicts
            "masks": None,      # Actual mask tensors
            "count": 0,         # Number of detected objects
            "image_size": (0, 0)  # (width, height)
        }
        
        logger.info("SatelliteVQAAgent initialized")

    def _reset_state(self):
        """Clear detection state between queries."""
        self.sam_state = {
            "objects": [], 
            "masks": None, 
            "count": 0, 
            "image_size": (0, 0)
        }

    def _detect_objects_wrapper(self, target_class: str) -> str:
        """
        Wrapper for SAM detection - stores results in state.
        Returns human-readable confirmation with available attributes.
        """
        logger.info(f"[Tool: detect_objects] Target: '{target_class}'")
        
        result = self.sam.segment_image(self.image, target_class, gsd=self.current_gsd)
        
        if result and result.get("metadata"):
            # Store complete metadata in state
            self.sam_state["objects"] = result["metadata"]
            self.sam_state["masks"] = result["masks"]
            self.sam_state["count"] = result["count"]
            self.sam_state["image_size"] = result["image_size"]
            
            # Return clear confirmation
            return (
                f"Detected {result['count']} '{target_class}' object(s). "
                f"Indices: 0 to {result['count']-1}. "
                f"Available attributes: area, width, height, shape, orientation, confidence."
            )
        else:
            self._reset_state()
            return f"No '{target_class}' objects detected in the image."
            
            
    def _get_object_info_wrapper(
        self, 
        sort_attribute: str, 
        rank_index: int, 
        return_attribute: str = None
    ) -> str:
        """
        Wrapper for object selection - accesses metadata directly.
        """
        if not self.sam_state["objects"]:
            return "Error: No objects detected. Call 'detect_objects' first."
        
        logger.info(
            f"[Tool: get_object_info] Sort by: {sort_attribute}, "
            f"Rank: {rank_index}, Return: {return_attribute or sort_attribute}"
        )
        
        # Direct metadata access - no parsing needed
        selected_obj, value_str = select_object_by_rank(
            self.sam_state["objects"],
            sort_attribute,
            rank_index,
            return_attribute
        )
        
        return value_str

    def _measure_distance_wrapper(self, index_1: int, index_2: int) -> str:
        """
        Wrapper for distance measurement - passes GSD correctly.
        """
        if not self.sam_state["objects"]:
            return "Error: No objects detected. Call 'detect_objects' first."
        
        logger.info(f"[Tool: measure_distance] Between indices {index_1} and {index_2}")
        
        # Direct metadata access with GSD
        result = calculate_distance_by_indices(
            self.sam_state["objects"],
            index_1,
            index_2,
            gsd=self.current_gsd
        )
        
        if isinstance(result, str):  # Error message
            return result
        else:  # Numeric result
            return f"{result:.2f}"

    def _calculate_wrapper(self, expression: str) -> str:
        """
        Wrapper for calculator tool.
        """
        logger.info(f"[Tool: calculate] Expression: {expression}")
        
        result = calculator_tool(expression)
        
        if isinstance(result, str):  
            return result
        else:
            return f"{result:.2f}"
            

    def _format_system_prompt(self) -> str:
        """
        Generate system prompt with tool definitions and output rules.
        """
        tools_json = json.dumps(self.tools_schema, indent=2)
        
        prompt = f"""You are a Satellite Image Analysis Agent with access to detection and measurement tools.

        AVAILABLE TOOLS:
        {tools_json}
        
        WORKFLOW:
        1. First call 'detect_objects' to find all instances of the target class
        2. Then use 'get_object_info' to query specific attributes
        3. Use 'measure_distance' for spatial measurements between objects
        4. Use 'calculate' for mathematical operations
        
        CRITICAL OUTPUT RULES:
        1. After calling tools, respond with ONLY the final answer - no explanation
        2. Format based on question type:
           - NUMERIC: Just the number (e.g., "450.5" or "3")
           - BINARY: Just "Yes" or "No"
           - SEMANTIC: Just 1-2 words (e.g., "rectangular" or "north")
        3. Do NOT write sentences like "The answer is..." or "There are..."
        4. Do NOT add units or punctuation to the final answer
        
        EXAMPLE 1 - Numeric Question:
        User: "How many buildings are there?"
        Step 1: {{"tool": "detect_objects", "arguments": {{"target_class": "building"}}}}
        Observation: Detected 5 'building' object(s). Indices: 0 to 4.
        Final Answer: 5
        
        EXAMPLE 2 - Semantic Question:
        User: "What is the shape of the largest building?"
        Step 1: {{"tool": "detect_objects", "arguments": {{"target_class": "building"}}}}
        Observation: Detected 3 'building' object(s). Indices: 0 to 2.
        Step 2: {{"tool": "get_object_info", "arguments": {{"sort_attribute": "area", "rank_index": -1, "return_attribute": "shape"}}}}
        Observation: rectangular
        Final Answer: rectangular
        
        EXAMPLE 3 - Binary Question:
        User: "Is there a ship in the image?"
        Step 1: {{"tool": "detect_objects", "arguments": {{"target_class": "ship"}}}}
        Observation: Detected 1 'ship' object(s).
        Final Answer: Yes
        
        Remember: Tools have direct access to all metadata. You only need to specify WHICH attribute to retrieve, not HOW to find it."""

        return prompt
    

    def run(
        self, 
        image: Image.Image, 
        user_query: str, 
        gsd: float = 1.0, 
        max_steps: int = 6
    ) -> Dict[str, Any]:
        """
        Execute the agent loop to answer a VQA query.
        
        Args:
            image: PIL Image
            user_query: User's question
            gsd: Ground sampling distance (meters/pixel)
            max_steps: Maximum reasoning steps
        
        Returns:
            {
                "final_answer": str,
                "steps_taken": int,
                "error": str (if failed)
            }
        """
        self.image = image
        self.current_gsd = gsd
        self._reset_state()
        
        system_prompt = self._format_system_prompt()
        
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_query}
                ]
            }
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {user_query}")
        logger.info(f"GSD: {gsd} m/pixel")
        logger.info(f"{'='*60}")
        
        for step in range(max_steps):
            # Prepare inputs for VLM
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.1
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"\n[Step {step + 1}] Model Output:\n{output_text}")
            
            # Try to parse and execute tool call
            tool_result = self._parse_and_execute_tool(output_text)
            
            if tool_result:
                # Tool was called - add to conversation
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}]
                })
                
                observation = f"Observation: {tool_result['result']}"
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": observation}]
                })
                
                logger.info(f"[Observation] {tool_result['result']}")
            else:
                # No tool call detected - this is the final answer
                final_answer = self._extract_clean_answer(output_text)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"FINAL ANSWER: {final_answer}")
                logger.info(f"Steps Taken: {step + 1}")
                logger.info(f"{'='*60}\n")
                
                return {
                    "final_answer": final_answer,
                    "steps_taken": step + 1
                }
        
        # Max steps reached
        logger.warning("Max steps reached without final answer")
        return {
            "error": "Max steps reached without producing final answer",
            "steps_taken": max_steps
        }

    def _parse_and_execute_tool(self, text: str) -> Union[Dict, None]:
        """
        Parse tool call from model output and execute it.
        
        Returns:
            {"tool": name, "result": output} if tool call found, else None
        """
        try:
            # Look for JSON object in output
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group(0))
            tool_name = data.get("tool")
            arguments = data.get("arguments", {})
            
            if tool_name not in self.tool_map:
                logger.warning(f"Unknown tool: {tool_name}")
                return None
            
            # Execute tool
            func = self.tool_map[tool_name]
            result = func(**arguments)
            
            return {"tool": tool_name, "result": result}
            
        except Exception as e:
            logger.debug(f"Failed to parse/execute tool: {e}")
            return None
    
    def _extract_clean_answer(self, text: str) -> str:
        """
        Extract clean final answer from model output.
        Removes common prefixes and punctuation.
        """
        # Remove common prefixes
        prefixes = [
            "the answer is",
            "final answer:",
            "answer:",
            "result:",
            "there are",
            "there is",
            "it is",
            "yes,",
            "no,",
        ]
        
        clean = text.strip().lower()
        
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        
        # Remove trailing punctuation
        clean = clean.rstrip('.,;:!?')
        
        # Capitalize first letter for semantic answers
        if clean and not clean[0].isdigit():
            clean = clean[0].upper() + clean[1:]
        
        return clean


