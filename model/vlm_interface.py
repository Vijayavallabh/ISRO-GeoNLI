"""
Interface for Vision Language Model inference using transformers.
Matches the notebook's ask_qwen method.
"""

import base64
import torch
from io import BytesIO
from qwen_vl_utils import process_vision_info


class VLMInterface:
    """
    Handles VLM inference operations using transformers.
    Closely matches the notebook's ask_qwen implementation.
    """
    
    def __init__(self, vlm_model, vlm_processor, device="cuda"):
        """
        Args:
            vlm_model: The loaded transformers model
            vlm_processor: The model processor
            device: Device to run inference on
        """
        self.model = vlm_model
        self.processor = vlm_processor
        self.device = device
    
    def _pil_to_data_uri(self, pil_image):
        """Convert PIL Image to base64 data URI (same as notebook)."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0):
        """
        Query the VLM with an image and text prompt.
        Matches the notebook's ask_qwen method exactly.
        
        Args:
            image: PIL Image
            prompt: Text prompt/question
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if image is not None:
            image_uri = self._pil_to_data_uri(image)
            content.append({"type": "image", "image": image_uri})
            
        content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": content
        })

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt"
        
        }
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        
        if temperature > 0:
            generation_config["temperature"] = temperature
            generation_config["top_k"] = -1
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generation_config
            )


        input_length = inputs['input_ids'].shape[1]
        generated_ids = generated_ids[0, input_length:]
        
        generated_text = self.processor.decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

