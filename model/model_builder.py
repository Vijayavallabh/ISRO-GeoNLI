"""
Model initialization using transformers.
For use with Qwen3-VL models like Qwen3-VL-8B.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from transformers import Sam3Model, Sam3Processor
from peft import PeftModel
import os

from dotenv import load_dotenv
load_dotenv()

def build_vlm_model(
    model_id="Dinosaur2314/qwen_finetune11", 
    device="cuda",
    torch_dtype=None,
    device_map="auto"
):
    """
    Build and initialize the Vision Language Model.
    Loads Qwen Base Model + Your LoRA Adapter.
    """
    
    # 1. Securely retrieve token
    hf_token = os.getenv('dino_hf_token')
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env. Private models may fail to load.")

    # 2. Define Base Model
    base_model_id = "Qwen/Qwen3-VL-8B-Instruct"

    print(f"Loading Processor from Base: {base_model_id}...")
    processor = AutoProcessor.from_pretrained(
        base_model_id, 
        trust_remote_code=True,
        token=hf_token,
        min_pixels = 256*256,
        max_pixels = 2048*2048,
        padding_side = "left"
    )
    
    # Determine torch dtype
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading Base Model Weights: {base_model_id}...")

    # 3. Load the Base Model first
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id, 
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True, 
            dtype=torch_dtype,
            token=hf_token
        )
    except Exception as e1:
        # Fallback to AutoModelForVision2Seq
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map if device_map else device,
                token=hf_token
            )
            print(f"Loaded as AutoModelForVision2Seq")
        except Exception as e2:
            raise RuntimeError(f"Failed to load Base Model: {e1}")
    
    # 4. Load and Apply your Fine-Tuned Adapter
    print(f"Loading LoRA Adapter: {model_id}...")
    try:
        model = PeftModel.from_pretrained(
            model, 
            model_id, ## Change to local path having weights
            token=hf_token
        )
    except Exception as e:
        print("Adapter load failed")
        raise RuntimeError(f"Could not load adapter.")

    # Set to eval mode
    model.eval()
    
    print(f"VLM (Base + Adapter) loaded successfully on {device}")
    return model, processor


def build_sam3_model(
    model_id="facebook/sam3",
    device="cuda"
):
    """
    Build and initialize SAM 3 segmentation model.
    """
    print(f"Loading SAM 3: {model_id}...")

    hf_token = os.getenv('sam_hf_token')
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found. Check .env file")
    
    processor = Sam3Processor.from_pretrained(model_id, token=hf_token)
    model = Sam3Model.from_pretrained(model_id, token=hf_token).to(device).eval()
    
    print(f"SAM 3 loaded successfully on {device}")
    return model, processor
