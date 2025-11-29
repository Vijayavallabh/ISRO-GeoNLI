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
            
        # Convert PIL image to data URI (same as notebook)
        image_uri = self._pil_to_data_uri(image)
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ]
        })
        
        # Prepare inputs using processor (same as notebook)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info (same as notebook - needed for Qwen models)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        # For transformers, use the processor to prepare inputs
        # The processor handles both text tokenization and image processing
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate with same parameters as notebook (temperature=0, deterministic)
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
        
        # Decode the generated text
        # Only decode the newly generated tokens (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        generated_ids = generated_ids[0, input_length:]
        
        generated_text = self.processor.decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

