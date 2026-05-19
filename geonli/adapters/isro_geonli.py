"""
Adapter layer that wraps the existing ISRO-GeoNLI monolithic code
and exposes it through the new ``geonli`` abstract interfaces.

This means users can continue using their current models/prompts
while gaining config-driven execution and CLI tools.
"""

from typing import Any, Dict, List, Optional
from PIL import Image

try:
    from rs_pipeline import RSPipeline
    from model.vlm_interface import VLMInterface
    from model.sam3_interface import SAM3Interface
    from tasks.captioning import CaptioningTask
    from tasks.grounding import GroundingTask
    from tasks.vqa import VQATask
    ISRO_AVAILABLE = True
except Exception as e:
    ISRO_AVAILABLE = False
    _IMPORT_ERR = e

from geonli.core.base import (
    VLMBase,
    TransformersVLMBase,
    SegmenterBase,
    TaskBase,
    TaskResult,
    SegmentationResult,
    Detection,
)
from geonli.core.registry import register_vlm, register_segmenter, register_task


# ---------------------------------------------------------------------------
# VLM Adapter
# ---------------------------------------------------------------------------

@register_vlm("isro-qwen3-vl")
class ISRO_VLM(TransformersVLMBase):
    """
    Wraps the existing ``VLMInterface`` (Qwen3-VL + LoRA).
    Inherits from TransformersVLMBase so that the full agent/router pipeline
    (tool-calling, chat_generate) works out of the box.
    """

    def __init__(
        self,
        vlm_model_id: str = "Dinosaur2314/qwen_finetune11",
        device: str = "cuda",
        **kwargs,
    ):
        if not ISRO_AVAILABLE:
            raise RuntimeError(
                f"ISRO-GeoNLI code not importable. "
                f"Ensure you are in the repo root. Error: {_IMPORT_ERR}"
            )
        self._vlm_model_id = vlm_model_id
        self._device = device
        self.model = None          # populated on _init
        self.processor = None      # populated on _init
        self._backend: Optional[VLMInterface] = None

    def _init(self):
        if self._backend is None:
            from model.model_builder import build_vlm_model
            m, p = build_vlm_model(self._vlm_model_id, self._device)
            self.model = m
            self.processor = p
            self._backend = VLMInterface(m, p, self._device)

    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0.0, **kwargs) -> str:
        self._init()
        return self._backend.query(
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Implement TransformersVLMBase.chat_generate using Qwen3-VL patterns.
        This is the primitive required by LLMRouter and TransformersSatelliteAgent.
        """
        self._init()
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = -1

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[0, input_length:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

    def model_name(self) -> str:
        return f"isro-qwen3-vl:{self._vlm_model_id}"


# ---------------------------------------------------------------------------
# Segmenter Adapter
# ---------------------------------------------------------------------------

@register_segmenter("isro-sam3")
class ISRO_Segmenter(SegmenterBase):
    """
    Wraps the existing ``SAM3Interface``.
    """

    def __init__(
        self,
        sam_model_id: str = "facebook/sam3",
        device: str = "cuda",
        spatial_resolution_m: float = 1.0,
        **kwargs,
    ):
        if not ISRO_AVAILABLE:
            raise RuntimeError(
                f"ISRO-GeoNLI code not importable. "
                f"Ensure you are in the repo root. Error: {_IMPORT_ERR}"
            )
        self._sam_model_id = sam_model_id
        self._device = device
        self._gsd = spatial_resolution_m
        self._backend: Optional[SAM3Interface] = None

    def _init(self):
        if self._backend is None:
            from model.model_builder import build_sam3_model
            m, p = build_sam3_model(self._sam_model_id, self._device)
            self._backend = SAM3Interface(m, p, self._device, self._gsd)

    def segment(self, image, text_prompt, **kwargs) -> Optional[SegmentationResult]:
        self._init()
        gsd = kwargs.get("gsd", self._gsd)
        result = self._backend.segment_image(image, text_prompt, gsd=gsd)
        if result is None:
            return None
        return SegmentationResult(
            masks=result.get("masks", []),
            metadata=result.get("metadata", []),
            count=result.get("count", 0),
            image_size=result.get("image_size", (0, 0)),
        )

    def model_name(self) -> str:
        return f"isro-sam3:{self._sam_model_id}"


# ---------------------------------------------------------------------------
# Task Adapters (thin wrappers)
# ---------------------------------------------------------------------------

@register_task("isro-captioning")
class ISROCaptioningTask(TaskBase):
    name = "captioning"

    def __init__(self, vlm: VLMBase, **kwargs):
        if not ISRO_AVAILABLE:
            raise RuntimeError(f"ISRO code not available: {_IMPORT_ERR}")
        self._vlm = vlm
        self._native_task: Optional[CaptioningTask] = None

    def _init(self):
        if self._native_task is None:
            if isinstance(self._vlm, ISRO_VLM):
                self._vlm._init()
                self._native_task = CaptioningTask(self._vlm._backend)
            else:
                raise TypeError("ISROCaptioningTask requires an ISRO_VLM instance.")

    def run(self, image, query, context=None) -> TaskResult:
        self._init()
        caption = self._native_task.generate_caption(image, query)
        return TaskResult(
            task_name=self.name,
            query=query,
            response=caption,
            metadata={"model": self._vlm.model_name()},
        )


@register_task("isro-grounding")
class ISROGroundingTask(TaskBase):
    name = "grounding"

    def __init__(self, vlm: VLMBase, segmenter: SegmenterBase, **kwargs):
        if not ISRO_AVAILABLE:
            raise RuntimeError(f"ISRO code not available: {_IMPORT_ERR}")
        self._vlm = vlm
        self._segmenter = segmenter
        self._native_task: Optional[GroundingTask] = None

    def _init(self):
        if self._native_task is None:
            if isinstance(self._vlm, ISRO_VLM) and isinstance(self._segmenter, ISRO_Segmenter):
                self._vlm._init()
                self._segmenter._init()
                self._native_task = GroundingTask(self._vlm._backend, self._segmenter._backend)
            else:
                raise TypeError("ISROGroundingTask requires ISRO_VLM + ISRO_Segmenter.")

    def run(self, image, query, context=None) -> TaskResult:
        self._init()
        dets = self._native_task.ground_objects(image, query, show_visualization=False)
        detections = [
            Detection(
                object_id=d.get("object-id", str(i)),
                obbox=d.get("obbox", []),
                score=1.0,
            )
            for i, d in enumerate(dets)
        ]
        return TaskResult(
            task_name=self.name,
            query=query,
            response=detections,
            metadata={"count": len(detections)},
        )


@register_task("isro-vqa")
class ISROVQATask(TaskBase):
    name = "vqa"

    def __init__(self, vlm: VLMBase, segmenter: SegmenterBase, **kwargs):
        if not ISRO_AVAILABLE:
            raise RuntimeError(f"ISRO code not available: {_IMPORT_ERR}")
        self._vlm = vlm
        self._segmenter = segmenter
        self._native_task: Optional[VQATask] = None

    def _init(self):
        if self._native_task is None:
            if isinstance(self._vlm, ISRO_VLM) and isinstance(self._segmenter, ISRO_Segmenter):
                self._vlm._init()
                self._segmenter._init()
                temp_gtask = GroundingTask(self._vlm._backend, self._segmenter._backend)
                self._native_task = VQATask(self._vlm._backend, temp_gtask, self._segmenter._backend)
            else:
                raise TypeError("ISROVQATask requires ISRO_VLM + ISRO_Segmenter.")

    def run(self, image, query, context=None) -> TaskResult:
        self._init()
        context = context or {}
        qtype = context.get("question_type", "semantic")
        gsd = context.get("metadata", {}).get("spatial_resolution_m", 1.0)
        answer = self._native_task.answer_question(image, query, gsd=gsd, question_type=qtype)
        return TaskResult(
            task_name=self.name,
            query=query,
            response=answer,
            metadata={"question_type": qtype},
        )
