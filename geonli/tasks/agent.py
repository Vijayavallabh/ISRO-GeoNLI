"""
Generic and Transformers-specific multi-step tool-calling agents.

The ``TransformersSatelliteAgent`` loads its system prompt from the
external prompt bank (template ``satellite_agent``), making it fully
customizable without touching Python code.
"""

import json
import logging
import re
from typing import Any, Dict, List, Union
from PIL import Image

from geonli.core.base import AgentBase, SegmenterBase, TransformersVLMBase
from geonli.core.registry import get_prompt
from geonli.utils.satellite_tools import (
    calculator_tool,
    calculate_distance_by_indices,
    select_object_by_rank,
)

logger = logging.getLogger(__name__)

SATELLITE_TOOLS_SCHEMA = [
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
                    "description": "Object class to detect (e.g., 'building', 'vehicle', 'tree', 'ship')",
                }
            },
            "required": ["target_class"],
        },
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
                    "description": "Attribute to sort objects by before selecting",
                },
                "rank_index": {
                    "type": "integer",
                    "description": (
                        "Which ranked object to select: "
                        "1 = smallest/first, -1 = largest/last, "
                        "2 = second smallest, -2 = second largest"
                    ),
                },
                "return_attribute": {
                    "type": "string",
                    "enum": ["area", "width", "height", "shape", "orientation", "confidence"],
                    "description": "The attribute to return from the selected object. If omitted, returns the sort_attribute value.",
                },
            },
            "required": ["sort_attribute", "rank_index"],
        },
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
                "index_1": {"type": "integer", "description": "Index of first object (0-based)"},
                "index_2": {"type": "integer", "description": "Index of second object (0-based)"},
            },
            "required": ["index_1", "index_2"],
        },
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
                "expression": {"type": "string", "description": "Math expression to evaluate (e.g., '(100 + 200) / 2')"},
            },
            "required": ["expression"],
        },
    },
]


class TransformersSatelliteAgent(AgentBase):
    """
    Multi-step tool-calling agent for transformers VLMs.
    Loads the system prompt from the external ``satellite_agent`` template.
    """

    def __init__(
        self,
        vlm: TransformersVLMBase,
        segmenter: SegmenterBase,
        prompt_template: str = "satellite_agent",
    ):
        self.vlm = vlm
        self.sam = segmenter
        self.prompt_template = prompt_template
        self.tools_schema = SATELLITE_TOOLS_SCHEMA

        self.tool_map = {
            "detect_objects": self._detect_objects_wrapper,
            "get_object_info": self._get_object_info_wrapper,
            "measure_distance": self._measure_distance_wrapper,
            "calculate": self._calculate_wrapper,
        }

        self.image: Image.Image = None
        self.current_gsd = 1.0
        self.sam_state: Dict[str, Any] = {
            "objects": [],
            "masks": None,
            "count": 0,
            "image_size": (0, 0),
        }

    def _reset_state(self):
        self.sam_state = {"objects": [], "masks": None, "count": 0, "image_size": (0, 0)}

    # -- Tool wrappers -------------------------------------------------

    def _detect_objects_wrapper(self, target_class: str) -> str:
        logger.info(f"[Tool: detect_objects] Target: '{target_class}'")
        result = self.sam.segment(self.image, target_class, gsd=self.current_gsd)

        if result and result.metadata:
            self.sam_state["objects"] = result.metadata
            self.sam_state["masks"] = result.masks
            self.sam_state["count"] = result.count
            self.sam_state["image_size"] = result.image_size
            return (
                f"Detected {result.count} '{target_class}' object(s). "
                f"Indices: 0 to {result.count - 1}. "
                f"Available attributes: area, width, height, shape, orientation, confidence."
            )
        else:
            self._reset_state()
            return f"No '{target_class}' objects detected in the image."

    def _get_object_info_wrapper(
        self, sort_attribute: str, rank_index: int, return_attribute: str = None
    ) -> str:
        if not self.sam_state["objects"]:
            return "Error: No objects detected. Call 'detect_objects' first."

        logger.info(
            f"[Tool: get_object_info] Sort by: {sort_attribute}, "
            f"Rank: {rank_index}, Return: {return_attribute or sort_attribute}"
        )
        _, value_str = select_object_by_rank(
            self.sam_state["objects"],
            sort_attribute,
            rank_index,
            return_attribute,
        )
        return value_str

    def _measure_distance_wrapper(self, index_1: int, index_2: int) -> str:
        if not self.sam_state["objects"]:
            return "Error: No objects detected. Call 'detect_objects' first."

        logger.info(f"[Tool: measure_distance] Between indices {index_1} and {index_2}")
        result = calculate_distance_by_indices(
            self.sam_state["objects"],
            index_1,
            index_2,
            gsd=self.current_gsd,
        )
        if isinstance(result, str):
            return result
        return f"{result:.2f}"

    def _calculate_wrapper(self, expression: str) -> str:
        logger.info(f"[Tool: calculate] Expression: {expression}")
        result = calculator_tool(expression)
        if isinstance(result, str):
            return result
        return f"{result:.2f}"

    # -- System prompt builder -----------------------------------------

    def _format_system_prompt(self) -> str:
        tools_json = json.dumps(self.tools_schema, indent=2)
        raw_template = get_prompt(self.prompt_template)
        return raw_template.format(tools_json=tools_json)

    # -- Main loop -----------------------------------------------------

    def run(
        self,
        image: Image.Image,
        user_query: str,
        gsd: float = 1.0,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        self.image = image
        self.current_gsd = gsd
        self._reset_state()

        system_prompt = self._format_system_prompt()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_query}]},
        ]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"QUERY: {user_query}")
        logger.info(f"GSD: {gsd} m/pixel")
        logger.info(f"{'=' * 60}")

        for step in range(max_steps):
            output_text = self.vlm.chat_generate(
                messages=messages,
                max_new_tokens=512,
                temperature=0.1,
            )

            logger.info(f"\n[Step {step + 1}] Model Output:\n{output_text}")

            tool_result = self._parse_and_execute_tool(output_text)

            if tool_result:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                })
                observation = f"Observation: {tool_result['result']}"
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": observation}],
                })
                logger.info(f"[Observation] {tool_result['result']}")
            else:
                final_answer = self._extract_clean_answer(output_text)
                logger.info(f"\n{'=' * 60}")
                logger.info(f"FINAL ANSWER: {final_answer}")
                logger.info(f"Steps Taken: {step + 1}")
                logger.info(f"{'=' * 60}\n")
                return {"final_answer": final_answer, "steps_taken": step + 1}

        logger.warning("Max steps reached without final answer")
        return {"error": "Max steps reached without producing final answer", "steps_taken": max_steps}

    def _parse_and_execute_tool(self, text: str) -> Union[Dict, None]:
        try:
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
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

    def _extract_clean_answer(self, text: str) -> str:
        prefixes = [
            "the answer is", "final answer:", "answer:", "result:",
            "there are", "there is", "it is", "yes,", "no,",
        ]
        clean = text.strip().lower()
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        clean = clean.rstrip(".,;:!?")
        if clean and not clean[0].isdigit():
            clean = clean[0].upper() + clean[1:]
        return clean
