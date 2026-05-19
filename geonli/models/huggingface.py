"""
Generic HuggingFace Vision-Language Model wrapper.
Auto-detects model family (Qwen, LLaVA, InternVL, generic) and uses the
appropriate preprocessing pipeline so users only provide a ``model_id``.

Supports both ``.query()`` (single-turn) and ``.chat_generate()`` (multi-turn,
required by the tool-calling agent) for any HF model with a chat template.
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image

from geonli.core.base import TransformersVLMBase
from geonli.core.registry import register_vlm


def _detect_architecture(model_id: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.lower()
    m = model_id.lower()
    if "qwen" in m:
        return "qwen"
    if "llava" in m:
        return "llava"
    if "internvl" in m:
        return "internvl"
    if "idefics" in m:
        return "idefics"
    return "generic"


@register_vlm("huggingface")
class HuggingFaceVLM(TransformersVLMBase):
    """
    Universal HF VLM wrapper.

    Args:
        model_id: HuggingFace model identifier (e.g. ``Qwen/Qwen3-VL-8B-Instruct``).
        device: ``"cuda"`` or ``"cpu"``.
        torch_dtype: ``"float16"``, ``"bfloat16"``, ``"float32"``, or ``"auto"``.
        architecture: Override auto-detection (``"qwen"``, ``"llava"``, ``"internvl"``, ``"generic"``).
        trust_remote_code: Passed to ``from_pretrained``.
        hf_token_env: Name of environment variable holding the HF token (default ``HF_TOKEN``).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        architecture: Optional[str] = None,
        trust_remote_code: bool = True,
        hf_token_env: str = "HF_TOKEN",
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.arch = _detect_architecture(model_id, architecture)
        self.trust_remote_code = trust_remote_code
        self._hf_token = os.getenv(hf_token_env) or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.attn_implementation = attn_implementation or "eager"

        # These are populated lazily
        self.model = None
        self.processor = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._is_loaded:
            return

        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, "auto")

        auth = {}
        if self._hf_token:
            auth["token"] = self._hf_token

        print(f"[HuggingFaceVLM] Loading {self.model_id} (family={self.arch})...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            **auth,
        )

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            **auth,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        # Resolve device_map (handle heterogeneous multi-GPU setups)
        device_map, max_memory = self._resolve_device_map()
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        try:
            self.model = AutoVLM.from_pretrained(self.model_id, **load_kwargs)
        except RuntimeError as e:
            if "FlashAttention" in str(e):
                print(f"[HuggingFaceVLM] FlashAttention failed, retrying with eager attention...")
                load_kwargs["attn_implementation"] = "eager"
                self.model = AutoVLM.from_pretrained(self.model_id, **load_kwargs)
            else:
                raise

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self._is_loaded = True
        print(f"[HuggingFaceVLM] Loaded on {self.device}.")

    def _resolve_device_map(self):
        import torch
        if self.device == "cpu":
            return None, None
        if self.device.startswith("cuda:"):
            return {"": self.device}, None
        # Default to a single large GPU for stability; users can override via
        # config extra: {device_map: "auto"} if they want multi-GPU sharding.
        return {"": "cuda:0"}, None

    # ------------------------------------------------------------------
    # VLMBase.query() — single-turn
    # ------------------------------------------------------------------

    def query(
        self,
        image: Optional[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        self._load()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        return self.chat_generate(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # TransformersVLMBase.chat_generate() — multi-turn
    # ------------------------------------------------------------------

    def chat_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        self._load()

        # Dispatch to family-specific preprocessing
        if self.arch == "qwen":
            return self._generate_qwen(messages, max_new_tokens, temperature)
        elif self.arch == "llava":
            return self._generate_llava(messages, max_new_tokens, temperature)
        elif self.arch == "internvl":
            return self._generate_internvl(messages, max_new_tokens, temperature)
        elif self.arch == "idefics":
            return self._generate_idefics(messages, max_new_tokens, temperature)
        else:
            return self._generate_generic(messages, max_new_tokens, temperature)

    # ------------------------------------------------------------------
    # Family-specific generators
    # ------------------------------------------------------------------

    def _generate_qwen(self, messages, max_new_tokens, temperature):
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        proc_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None:
            proc_kwargs["images"] = image_inputs
        if video_inputs is not None:
            proc_kwargs["videos"] = video_inputs

        inputs = self.processor(**proc_kwargs)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        if temperature == 0.0:
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        else:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
            }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[0, input_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_llava(self, messages, max_new_tokens, temperature):
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract images from messages
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)

        if images:
            inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim prompt tokens
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _generate_internvl(self, messages, max_new_tokens, temperature):
        # InternVL uses a similar pattern but its processor expects different kwargs.
        # Fallback to generic first; can be refined when InternVL-specific needs arise.
        return self._generate_generic(messages, max_new_tokens, temperature)

    def _generate_idefics(self, messages, max_new_tokens, temperature):
        # Idefics uses its own chat format; fallback to generic.
        return self._generate_generic(messages, max_new_tokens, temperature)

    def _generate_generic(self, messages, max_new_tokens, temperature):
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Try to extract images
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)

        try:
            if images:
                inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(text=text, return_tensors="pt", padding=True)
        except TypeError:
            # Some processors don't accept `images=` (e.g., older ones)
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def model_name(self) -> str:
        return f"huggingface:{self.arch}:{self.model_id}"
