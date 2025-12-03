"""
Main pipeline orchestrator for Remote Sensing tasks.
Uses transformers backend for Qwen3-VL models.
"""

import torch
from model.model_builder import build_vlm_model, build_sam3_model
from model.vlm_interface import VLMInterface
from model.sam3_interface import SAM3Interface
from tasks.captioning import CaptioningTask
from tasks.grounding import GroundingTask
from tasks.vqa import VQATask


class RSPipeline:
    """
    Main pipeline for Remote Sensing image analysis tasks.
    
    Supports:
    - Image captioning
    - Object grounding/detection
    - Visual Question Answering (numeric, binary, semantic)
    """
    
    def __init__(
        self,
        vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
        sam_model_id="facebook/sam3",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the pipeline with models.
        
        Args:
            vlm_model_id: HuggingFace model ID for VLM (default: Qwen/Qwen3-VL-8B)
            sam_model_id: HuggingFace model ID for SAM 3
            device: Device to run models on
        """
        self.device = device
        
        print(f"--- Initializing RS Pipeline on {self.device} ---")
        
        # Build VLM model (transformers only)
        vlm_model, vlm_processor = build_vlm_model(vlm_model_id, device)
        self.vlm = VLMInterface(vlm_model, vlm_processor, device)
        
        # Build SAM 3
        sam_model, sam_processor = build_sam3_model(sam_model_id, device)
        self.sam3 = SAM3Interface(sam_model, sam_processor, device)
        
        # Initialize task handlers
        self.captioning = CaptioningTask(self.vlm)
        self.grounding = GroundingTask(self.vlm, self.sam3)
        self.vqa = VQATask(self.vlm, self.grounding)
        
        print("Pipeline Initialization Complete.")
    
    def generate_caption(self, image, instruction):
        """
        Generate image caption.
        
        Args:
            image: PIL Image
            instruction: Captioning instruction
            
        Returns:
            Caption string
        """
        return self.captioning.generate_caption(image, instruction)
    
    def ground_objects(self, image, query, gsd=1.0, score_threshold=0.4):
        """
        Detect and localize objects in image.
        
        Args:
            image: PIL Image
            query: Object description/query
            gsd: Ground Sample Distance (meters per pixel)
            score_threshold: Confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        return self.grounding.ground_objects(
            image, query, gsd, score_threshold
        )
    
    def answer_question(self, image, query, detections=None, 
                       question_type="numeric", gsd=1.0):
        """
        Answer a question about the image using the Smart Router.
        
        Args:
            image: PIL Image
            query: Question string
            detections: Existing detections (optional)
            question_type: DEPRECATED (Router handles this now)
            gsd: Ground Sample Distance
            
        Returns:
            Answer string
        """
        if detections is None:
            detections = []
        return self.vqa.answer_question(image, query, detections, gsd)
