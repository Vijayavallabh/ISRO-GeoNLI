"""
Custom VLM loader for direct Hugging Face model loading
Replaces vLLM server with direct model inference
"""

import torch
import re
import os
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from typing import Dict, List, Any

class DirectVLMLoader:
    """Load and run VLM directly from Hugging Face"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Thinking", device: str = "cuda"):
        print(f"Loading VLM model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"✓ Model loaded successfully on {device}")
    
    def generate(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        # Extract image objects and prepare messages for chat template
        processed_messages = []
        images = []
        
        for msg in messages:
            new_content = []
            if isinstance(msg['content'], str):
                new_content.append({"type": "text", "text": msg['content']})
            elif isinstance(msg['content'], list):
                for item in msg['content']:
                    if item.get('type') == 'text':
                        new_content.append(item)
                    elif item.get('type') == 'image_url':
                        image_url = item['image_url']['url']
                        # Handle file:// prefix
                        path = image_url[7:] if image_url.startswith('file://') else image_url
                        
                        try:
                            # Only add to images list if file exists
                            if os.path.exists(path):
                                img = Image.open(path).convert('RGB')
                                images.append(img)
                                # Add placeholder for chat template
                                new_content.append({"type": "image", "image": path})
                                print(f"DEBUG: Loaded image from {path}")
                            else:
                                print(f"WARNING: Image path does not exist: {path}")
                        except Exception as e:
                            print(f"WARNING: Could not load image {path}: {e}")
            
            processed_messages.append({"role": msg["role"], "content": new_content})

        # Apply chat template
        text = self.processor.apply_chat_template(
            processed_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Prepare inputs dynamically to avoid passing empty lists/None
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt"
        }
        
        # CRITICAL FIX: Only pass 'images' arg if we actually have images.
        # Passing [] or None explicitly can crash Qwen processors.
        if images and len(images) > 0:
            processor_kwargs["images"] = images
            
        inputs = self.processor(**processor_kwargs).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False, 
                temperature=0.0
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # CLEANUP: Remove <think> tags if present
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
        
        return generated_text


def send_generate_request_custom(
    vlm_loader: DirectVLMLoader,
    messages: List[Dict],
    max_tokens: int = 1000,
    **kwargs
) -> Dict:
    try:
        text = vlm_loader.generate(messages, max_tokens)
        
        response = {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': text
                },
                'finish_reason': 'stop'
            }]
        }
        return response
        
    except Exception as e:
        print(f"Error in VLM generation: {e}")
        raise