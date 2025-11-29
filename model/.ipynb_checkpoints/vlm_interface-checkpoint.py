"""
Interface for Vision Language Model inference.
"""

import base64
from io import BytesIO
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info


class VLMInterface:
    """Handles VLM inference operations."""
    
    def __init__(self, vlm_model, vlm_processor):
        """
        Args:
            vlm_model: The loaded vLLM model
            vlm_processor: The model processor
        """
        self.model = vlm_model
        self.processor = vlm_processor
    
    def _pil_to_data_uri(self, pil_image):
        """Convert PIL Image to base64 data URI."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def query(self, image, prompt, system_prompt=None, max_tokens=512, temperature=0):
        """
        Query the VLM with an image and text prompt.
        
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
            
        # Convert PIL image to data URI for vLLM
        image_uri = self._pil_to_data_uri(image)
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ]
        })
        
        # Prepare inputs for vLLM
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        
        inputs = {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=-1,
            stop_token_ids=[]
        )
        
        outputs = self.model.generate([inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
