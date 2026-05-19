"""
Generic captioning task.
Uses the PromptManager so system prompts live outside Python code.
Matches the original ISRO-GeoNLI behavior exactly.
"""

from typing import Any, Dict, Optional
from PIL import Image

from geonli.core.base import TaskBase, TaskResult, VLMBase
from geonli.core.registry import register_task, get_prompt


@register_task("captioning")
class CaptioningTask(TaskBase):
    """
    Generates a detailed but concise caption using a single VLM call
    guided by a system prompt (~70 words, factual, no speculation).
    """
    name = "captioning"

    def __init__(
        self,
        vlm: VLMBase,
        prompt_template: str = "default_caption",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.vlm = vlm
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(
        self,
        image: Image.Image,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        # Load system prompt from template bank (exact original prompt)
        system_prompt = None
        try:
            system_prompt = get_prompt(self.prompt_template)
        except KeyError:
            system_prompt = None

        final_caption = self.vlm.query(
            image=image,
            prompt=query,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return TaskResult(
            task_name=self.name,
            query=query,
            response=final_caption.strip(),
            metadata={"model": self.vlm.model_name()},
        )
