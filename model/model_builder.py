"""
Model initialization using transformers.
For use with Qwen3-VL models like Qwen3-VL-8B.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Sam3Model, Sam3Processor
from qwen_vl_utils import process_vision_info
import os

from dotenv import load_dotenv
load_dotenv()

def build_vlm_model(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype=None,
    device_map="auto"
):
    """
    Build and initialize the Vision Language Model using transformers.
    
    This matches the notebook's approach but uses standard transformers API.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen3-VL-8B")
        device: Device to load model on
        torch_dtype: Data type for model (None for auto, torch.float16 for FP16)
        device_map: Device mapping strategy ("auto", "cuda", etc.)
        
    Returns:
        tuple: (vlm_model, vlm_processor)
    """
    print(f"Loading VLM: {model_id}...")
    
    # Load processor (same as notebook)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Determine torch dtype
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.float16  # Use FP16 for efficiency
        else:
            torch_dtype = torch.float32
    
    # Try AutoModelForCausalLM first (most common for Qwen models)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto",
        trust_remote_code=True, dtype=torch_dtype
        )

    except Exception as e1:
        # Fallback to AutoModelForVision2Seq
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map if device_map else device
            )
            print(f"Loaded as AutoModelForVision2Seq")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model with both AutoModelForCausalLM ({e1}) "
                f"and AutoModelForVision2Seq ({e2})"
            )
    
    # Set to eval mode
    model.eval()
    
    print(f"VLM loaded successfully on {device}")
    return model, processor


def build_sam3_model(
    model_id="facebook/sam3",
    device="cuda"
):
    """
    Build and initialize SAM 3 segmentation model.
    
    Args:
        model_id: HuggingFace model identifier for SAM 3
        device: Device to load model on
        
    Returns:
        tuple: (sam_model, sam_processor)
    """
    print(f"Loading SAM 3: {model_id}...")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found. Check .env file")
    
    processor = Sam3Processor.from_pretrained(model_id, token=hf_token)
    model = Sam3Model.from_pretrained(model_id, token=hf_token).to(device).eval()
    
    print(f"SAM 3 loaded successfully on {device}")
    return model, processor

