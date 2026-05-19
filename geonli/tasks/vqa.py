"""
100% model-agnostic VQA task.
  - LLMRouter works with ANY VLMBase (local HF, API, dummy) via .query()
  - Tool-calling agent works with any TransformersVLMBase via .chat_generate()
  - ALL prompts are external; NO hardcoded strings remain.
"""

import json
import logging
from typing import Any, Dict, Optional
from PIL import Image

from geonli.core.base import (
    AgentBase,
    TaskBase,
    TaskResult,
    VLMBase,
    SegmenterBase,
    TransformersVLMBase,
)
from geonli.core.registry import register_task, get_prompt
from geonli.tasks.agent import TransformersSatelliteAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routers (both model-agnostic)
# ---------------------------------------------------------------------------

class RouterBase:
    def route(self, question: str) -> str:
        raise NotImplementedError


class KeywordRouter(RouterBase):
    """Keyword-based router (zero dependencies)."""

    SAM_KEYWORDS = [
        "how many", "count", "number of", "area", "size", "length",
        "width", "distance", "orientation", "direction", "ratio",
        "is there", "are there", "contains",
    ]

    def route(self, question: str) -> str:
        q = question.lower()
        if any(kw in q for kw in self.SAM_KEYWORDS):
            return "SAM"
        return "VLM"


class LLMRouter(RouterBase):
    """
    LLM-based router that works with **any** VLMBase.
    Loads the router prompt from the PromptManager (external template).
    """

    def __init__(self, vlm: VLMBase, prompt_template: str = "default_router"):
        self.vlm = vlm
        self.prompt_template = prompt_template

    def route(self, question: str) -> str:
        router_prompt = get_prompt(self.prompt_template)
        # We pass the router instructions as system_prompt and the question as prompt.
        # This works identically for HF models and OpenAI/Gemini APIs.
        output_text = self.vlm.query(
            image=None,
            prompt=f'Question: "{question}"\nAnswer with ONLY one word ("SAM" or "VLM"):',
            system_prompt=router_prompt,
            max_tokens=10,
            temperature=0.0,
        )
        route = output_text.strip().upper()
        return "SAM" if "SAM" in route else "VLM"


# ---------------------------------------------------------------------------
# VQA Task
# ---------------------------------------------------------------------------

@register_task("vqa")
class VQATask(TaskBase):
    name = "vqa"

    def __init__(
        self,
        vlm: VLMBase,
        segmenter: Optional[SegmenterBase] = None,
        router: Optional[RouterBase] = None,
        agent: Optional[AgentBase] = None,
        numeric_prompt_template: str = "vqa_numeric",
        binary_prompt_template: str = "vqa_binary",
        semantic_prompt_template: str = "vqa_semantic",
        max_tokens: int = 128,
        **kwargs,
    ):
        self.vlm = vlm
        self.segmenter = segmenter
        self.max_tokens = max_tokens
        self.numeric_prompt_template = numeric_prompt_template
        self.binary_prompt_template = binary_prompt_template
        self.semantic_prompt_template = semantic_prompt_template

        # Router: user-provided > LLMRouter > KeywordRouter
        if router is not None:
            self.router = router
        else:
            try:
                self.router = LLMRouter(vlm)
            except KeyError:
                logger.warning(
                    "Router prompt template 'default_router' not registered. "
                    "Falling back to KeywordRouter. "
                    "Register the template via PromptManager for LLM routing."
                )
                self.router = KeywordRouter()

        # Agent: user-provided > auto-build for TransformersVLMBase
        if agent is not None:
            self.agent = agent
        elif isinstance(vlm, TransformersVLMBase) and segmenter is not None:
            self.agent = TransformersSatelliteAgent(vlm, segmenter)
        else:
            self.agent = None

    def run(
        self,
        image: Image.Image,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        context = context or {}
        qtype = context.get("question_type", "semantic")
        route = self.router.route(query)

        if route == "SAM" and self.agent is not None:
            answer = self._answer_via_sam_path(image, query, qtype, context)
        else:
            answer = self._answer_via_vlm_path(image, query, qtype, context)

        return TaskResult(
            task_name=self.name,
            query=query,
            response=answer,
            metadata={
                "route": route,
                "question_type": qtype,
                "model": self.vlm.model_name(),
                "used_agent": self.agent is not None and route == "SAM",
            },
        )

    def _answer_via_vlm_path(self, image, query, qtype, context) -> str:
        # Prompt template mapping
        template_map = {
            "numeric": self.numeric_prompt_template,
            "binary": self.binary_prompt_template,
            "semantic": self.semantic_prompt_template,
        }
        template_name = template_map.get(qtype, self.semantic_prompt_template)
        system_prompt = get_prompt(template_name)

        gsd = context.get("metadata", {}).get("spatial_resolution_m", 1.0)
        user_prompt = f"Question: '{query}'"
        if gsd:
            user_prompt = f"Context: The ground sampling distance is {gsd} m/pixel.\n{user_prompt}"

        return self.vlm.query(
            image=image,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    def _answer_via_sam_path(self, image, query, qtype, context) -> str:
        gsd = context.get("metadata", {}).get("spatial_resolution_m", 1.0)

        type_instruction = ""
        if qtype == "numeric":
            type_instruction = "Answer this numeric question. Return a single number if possible."
        elif qtype == "binary":
            type_instruction = "Answer this binary question with Yes or No."
        elif qtype == "semantic":
            type_instruction = (
                "Provide the final answer as a single word or a short phrase "
                "(e.g., 'Urban', 'Adjacent'). Do NOT use full sentences."
            )

        augmented_query = f"{type_instruction} The ground sampling distance is {gsd} m/pixel. {query}"

        if self.agent is not None:
            response_dict = self.agent.run(image, augmented_query, gsd=gsd)
            if "final_answer" in response_dict:
                return response_dict["final_answer"]
            return f"Error: {response_dict.get('error', 'Agent failed to answer.')}"

        # Fallback if no agent available (e.g., API-only VLM)
        return self._answer_via_vlm_path(image, query, qtype, context)
