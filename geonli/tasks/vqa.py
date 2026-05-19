"""
Generic VQA task with the FULL original ISRO-GeoNLI pipeline:
  - LLM-based router (transformers models) or keyword router (generic VLMs)
  - Tool-calling agent for SAM-routed questions (transformers models)
  - Specialized system prompts per question type for VLM-routed questions
Works with any VLMBase; automatically upgrades to full agent logic
when given a TransformersVLMBase.
"""

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
# Routers
# ---------------------------------------------------------------------------

class RouterBase:
    """Minimal router interface."""

    def route(self, question: str) -> str:
        """Return 'SAM' or 'VLM'."""
        raise NotImplementedError


class KeywordRouter(RouterBase):
    """Keyword-based router (works with any VLMBase)."""

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
    LLM-based router using the full original system prompt.
    Requires a TransformersVLMBase because it uses chat_generate().
    """

    def __init__(self, vlm: TransformersVLMBase, prompt_template: str = "default_router"):
        self.vlm = vlm
        self.prompt_template = prompt_template

    def route(self, question: str) -> str:
        try:
            system_prompt = get_prompt(self.prompt_template)
        except KeyError:
            system_prompt = self._default_router_prompt()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"Question: {question}\nAnswer:"}]},
        ]

        output_text = self.vlm.chat_generate(
            messages=messages,
            max_new_tokens=10,
            temperature=0.1,
        )
        route = output_text.strip().upper()
        return "SAM" if "SAM" in route else "VLM"

    def _default_router_prompt(self) -> str:
        return (
            "You are an intelligent routing system for remote sensing VQA.\n"
            'Respond with ONLY "SAM" if the question requires segmentation, '
            'or "VLM" if it can be answered by visual understanding alone.\n'
            'Output ONLY one word: "SAM" or "VLM".'
        )


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
        max_tokens: int = 128,
    ):
        self.vlm = vlm
        self.segmenter = segmenter
        self.max_tokens = max_tokens

        # Auto-select router based on VLM capabilities
        if router is not None:
            self.router = router
        elif isinstance(vlm, TransformersVLMBase):
            self.router = LLMRouter(vlm)
        else:
            self.router = KeywordRouter()

        # Auto-select agent based on VLM capabilities
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

    # -- VLM path -------------------------------------------------------

    def _answer_via_vlm_path(self, image, query, qtype, context) -> str:
        sys_prompt = self._system_prompt_for_type(qtype)
        gsd = context.get("metadata", {}).get("spatial_resolution_m", 1.0)
        user_prompt = f"Question: '{query}'"
        if gsd:
            user_prompt = f"Context: The ground sampling distance is {gsd} m/pixel.\n{user_prompt}"

        return self.vlm.query(
            image=image,
            prompt=user_prompt,
            system_prompt=sys_prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    @staticmethod
    def _system_prompt_for_type(qtype: str) -> Optional[str]:
        prompts = {
            "numeric": (
                "You are a remote sensing assistant. "
                "The user asks a numeric question, to be answered only with a numeric value. "
                "Estimate or count the required value based on the visual image, and all the relevant context from the question. "
                "Provide the number clearly."
            ),
            "binary": (
                "You are a remote sensing assistant. "
                "The user asks a binary (Yes/No) question. "
                "Analyze the image and the question context, and answer with ONLY 'Yes' or 'No'."
            ),
            "semantic": (
                "You are a remote sensing assistant. "
                "Answer the question directly and concisely using as few words as possible. "
                "Do not answer with full sentences. "
                "Example: 'Rectangular' instead of 'The field is rectangular'. "
                "Example: 'Blue' instead of 'It is blue'."
            ),
        }
        return prompts.get(qtype)

    # -- SAM path -------------------------------------------------------

    def _answer_via_sam_path(self, image, query, qtype, context) -> str:
        gsd = context.get("metadata", {}).get("spatial_resolution_m", 1.0)

        # Type-specific instruction prefix (exact original behavior)
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
