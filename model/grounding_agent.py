"""
Agentic mask selection for grounding pipeline.
Uses tool calling to intelligently select masks based on query requirements.
"""

import torch
import json
import re
from typing import List, Dict, Any, Union
from PIL import Image
from qwen_vl_utils import process_vision_info
from utils.grounding_selection_tools import (
    filter_masks_by_region,
    select_masks_by_attribute_rank,
    select_masks_by_attribute_threshold,
    select_all_masks,
    select_masks_by_count,
    calculate_mask_distance,
    get_mask_attribute
)

import logging

logger = logging.getLogger(__name__)


GROUNDING_TOOLS = [
    {
        "name": "select_all",
        "description": (
            "Selects ALL detected masks. Use when the query asks for multiple/all objects "
            "(e.g., 'all buildings', 'locate vehicles', 'find trees')."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "select_by_region",
        "description": (
            "Selects masks in a specific spatial region of the image. "
            "Use when query mentions location (e.g., 'top-left building', 'vehicles in the bottom')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "enum": ["top", "bottom", "left", "right", "center", 
                            "top-left", "top-right", "bottom-left", "bottom-right"],
                    "description": "Spatial region to filter masks"
                }
            },
            "required": ["region"]
        }
    },
    {
        "name": "select_by_rank",
        "description": (
            "Selects a single mask by ranking on an attribute. "
            "Use when query asks for superlatives (e.g., 'largest building', 'smallest vehicle', "
            "'most confident detection')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attribute": {
                    "type": "string",
                    "enum": ["area", "confidence"],
                    "description": "Attribute to rank by"
                },
                "rank": {
                    "type": "integer",
                    "description": "1=smallest/lowest, -1=largest/highest, 2=second smallest, etc."
                }
            },
            "required": ["attribute", "rank"]
        }
    },
    {
        "name": "select_top_n",
        "description": (
            "Selects top N masks by an attribute. "
            "Use when query specifies a count (e.g., 'top 3 largest buildings', '5 most confident detections')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of masks to select"
                },
                "attribute": {
                    "type": "string",
                    "enum": ["area", "confidence"],
                    "description": "Attribute to sort by"
                },
                "order": {
                    "type": "string",
                    "enum": ["largest", "smallest"],
                    "description": "Sort order"
                }
            },
            "required": ["count", "attribute", "order"]
        }
    },
    {
        "name": "select_by_threshold",
        "description": (
            "Selects masks meeting an attribute threshold. "
            "Use when query specifies conditions (e.g., 'buildings larger than 500 sq m', "
            "'high confidence detections')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attribute": {
                    "type": "string",
                    "enum": ["area", "confidence"],
                    "description": "Attribute to filter by"
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold value"
                },
                "comparison": {
                    "type": "string",
                    "enum": ["greater", "less"],
                    "description": "Comparison operator"
                }
            },
            "required": ["attribute", "threshold", "comparison"]
        }
    }
]


class GroundingSelectionAgent:
    """
    Tool-calling agent for intelligent mask selection in grounding pipeline.
    """
    
    def __init__(self, vlm_model, vlm_processor, device="cuda"):
        self.model = vlm_model
        self.processor = vlm_processor
        self.device = device
        self.tools_schema = GROUNDING_TOOLS
        
        # Map tool names to functions
        self.tool_map = {
            "select_all": self._select_all_wrapper,
            "select_by_region": self._select_by_region_wrapper,
            "select_by_rank": self._select_by_rank_wrapper,
            "select_top_n": self._select_top_n_wrapper,
            "select_by_threshold": self._select_by_threshold_wrapper,
        }
        
        # State
        self.masks_metadata = []
        self.image_size = (0, 0)
        self.annotated_image = None
        
        logger.info("GroundingSelectionAgent initialized")
    
    def _select_all_wrapper(self) -> str:
        """Select all masks."""
        logger.info("[Tool: select_all]")
        selected = select_all_masks(self.masks_metadata)
        return f"Selected {len(selected)} mask(s): {selected}"
    
    def _select_by_region_wrapper(self, region: str) -> str:
        """Select masks by spatial region."""
        logger.info(f"[Tool: select_by_region] Region: {region}")
        selected = filter_masks_by_region(
            self.masks_metadata,
            region,
            self.image_size
        )
        return f"Selected {len(selected)} mask(s) in region '{region}': {selected}"
    
    def _select_by_rank_wrapper(self, attribute: str, rank: int) -> str:
        """Select mask by attribute ranking."""
        logger.info(f"[Tool: select_by_rank] Attribute: {attribute}, Rank: {rank}")
        selected = select_masks_by_attribute_rank(
            self.masks_metadata,
            attribute,
            rank
        )
        if selected:
            return f"Selected mask {selected[0]} (rank {rank} by {attribute})"
        return "No mask found matching criteria"
    
    def _select_top_n_wrapper(self, count: int, attribute: str, order: str) -> str:
        """Select top N masks by attribute."""
        logger.info(f"[Tool: select_top_n] Count: {count}, Attribute: {attribute}, Order: {order}")
        selected = select_masks_by_count(
            self.masks_metadata,
            count,
            attribute,
            order
        )
        return f"Selected top {len(selected)} mask(s) by {order} {attribute}: {selected}"
    
    def _select_by_threshold_wrapper(
        self, 
        attribute: str, 
        threshold: float, 
        comparison: str
    ) -> str:
        """Select masks by attribute threshold."""
        logger.info(f"[Tool: select_by_threshold] {attribute} {comparison} {threshold}")
        selected = select_masks_by_attribute_threshold(
            self.masks_metadata,
            attribute,
            threshold,
            comparison
        )
        return f"Selected {len(selected)} mask(s) where {attribute} {comparison} {threshold}: {selected}"
    
    def _format_system_prompt(self) -> str:
        """Generate system prompt with tool definitions."""
        tools_json = json.dumps(self.tools_schema, indent=2)
        
        prompt = f"""You are a Mask Selection Agent for satellite/aerial image grounding tasks.

        AVAILABLE TOOLS:
        {tools_json}
        
        TASK:
        You have been given candidate masks detected by SAM3. Your job is to select the appropriate mask(s) 
        based on the user's query by calling ONE tool.
        
        METADATA AVAILABLE:
        Each mask has:
        - mask_id: Unique identifier (integer)
        - area: Area in square meters (float)
        - confidence: Detection confidence score (float, 0-1)
        - obb: 8-point oriented bounding box coordinates
        
        SELECTION STRATEGY:
        1. If query asks for ALL objects → use 'select_all'
        2. If query mentions LOCATION/REGION → use 'select_by_region'
        3. If query asks for LARGEST/SMALLEST/BEST → use 'select_by_rank'
        4. If query specifies a COUNT (top N) → use 'select_top_n'
        5. If query has CONDITIONS/THRESHOLDS → use 'select_by_threshold'
        
        OUTPUT FORMAT:
        After calling a tool, you will receive a list of selected mask IDs.
        Respond with ONLY those mask IDs in your final answer, nothing else.
        
        EXAMPLE 1:
        Query: "Locate all buildings in the image"
        Tool Call: {{"tool": "select_all", "arguments": {{}}}}
        Observation: Selected 5 mask(s): [0, 1, 2, 3, 4]
        Final Answer: [0, 1, 2, 3, 4]
        
        EXAMPLE 2:
        Query: "Find the largest building"
        Tool Call: {{"tool": "select_by_rank", "arguments": {{"attribute": "area", "rank": -1}}}}
        Observation: Selected mask 3 (rank -1 by area)
        Final Answer: [3]
        
        EXAMPLE 3:
        Query: "Locate vehicles in the top-left"
        Tool Call: {{"tool": "select_by_region", "arguments": {{"region": "top-left"}}}}
        Observation: Selected 2 mask(s) in region 'top-left': [0, 5]
        Final Answer: [0, 5]"""

        return prompt
    
    def select_masks(
        self,
        query: str,
        masks_metadata: List[Dict],
        image_size: tuple,
        annotated_image: Image.Image,
        max_steps: int = 3
    ) -> List[int]:
        """
        Select appropriate masks based on query using tool calling.
        
        Args:
            query: User's grounding query
            masks_metadata: List of mask metadata dicts
            image_size: (width, height) tuple
            annotated_image: Annotated image showing masks
            max_steps: Maximum reasoning steps
        
        Returns:
            List of selected mask_ids
        """
        self.masks_metadata = masks_metadata
        self.image_size = image_size
        self.annotated_image = annotated_image
        
        system_prompt = self._format_system_prompt()
        
        # Create metadata summary
        metadata_summary = "\n".join([
            f"Mask {m['mask_id']}: area={m['area']:.2f} sq.m, confidence={m['confidence']:.3f}"
            for m in masks_metadata
        ])
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": annotated_image},
                    {"type": "text", "text": f"QUERY: {query}\n\nDETECTED MASKS:\n{metadata_summary}\n\nSelect the appropriate mask(s)."}
                ]
            }
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"GROUNDING QUERY: {query}")
        logger.info(f"CANDIDATE MASKS: {len(masks_metadata)}")
        logger.info(f"{'='*60}")
        
        for step in range(max_steps):
            # Prepare inputs
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
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1
                )
            
            # Decode
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
                # Tool was called
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
                # No tool call - extract final answer
                selected_ids = self._extract_mask_ids(output_text)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"SELECTED MASKS: {selected_ids}")
                logger.info(f"Steps Taken: {step + 1}")
                logger.info(f"{'='*60}\n")
                
                return selected_ids
        
        # Fallback: return all masks if no decision made
        logger.warning("Max steps reached - selecting all masks as fallback")
        return [m['mask_id'] for m in masks_metadata]
    
    def _parse_and_execute_tool(self, text: str) -> Union[Dict, None]:
        """Parse and execute tool call from model output."""
        try:
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group(0))
            tool_name = data.get("tool")
            arguments = data.get("arguments", {})
            
            if tool_name not in self.tool_map:
                logger.warning(f"Unknown tool: {tool_name}")
                return None
            
            func = self.tool_map[tool_name]
            result = func(**arguments)
            
            return {"tool": tool_name, "result": result}
            
        except Exception as e:
            logger.debug(f"Failed to parse/execute tool: {e}")
            return None
    
    def _extract_mask_ids(self, text: str) -> List[int]:
        """Extract mask IDs from final answer."""
        # Look for list format: [0, 1, 2] or [0,1,2]
        list_match = re.search(r'\[([0-9,\s]+)\]', text)
        if list_match:
            ids_str = list_match.group(1)
            ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
            return ids
        
        # Look for individual numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return [int(n) for n in numbers]
        
        return []
