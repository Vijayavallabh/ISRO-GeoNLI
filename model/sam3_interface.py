
import torch
import numpy as np
import cv2
from PIL import Image
from utils.geo_calc import GeoCalculator


class SAM3Interface:
    
    def __init__(self, sam_model, sam_processor, device="cuda", spatial_resolution_m=1.0):
        self.model = sam_model
        self.processor = sam_processor
        self.device = device
        self.geo_calc = GeoCalculator(spatial_resolution_m=spatial_resolution_m)

    def segment_image(self, image: Image.Image, target_class):
        try:
            inputs = self.processor(
                images=image,
                text=[target_class],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            img_h, img_w = image.size[1], image.size[0]
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.3,
                mask_threshold=0.3,
                target_sizes=[(img_h, img_w)]
            )[0]
            
            masks = results.get("masks")
            scores = results.get("scores")
            
            if masks is None or len(masks) == 0:
                return None
                
            sam_metadata = []
            
            for idx, (mask, score) in enumerate(zip(masks, scores)):
                mask_np = mask.cpu().numpy()
                mask_bool = (mask_np > 0.5).astype(bool)
                
                geo_data = self.geo_calc.extract_metadata_from_mask(mask_bool)
                
                if geo_data is not None:
                    sam_metadata.append({
                        "mask_id": idx,
                        "confidence": float(score.cpu().numpy()),
                        **geo_data
                    })
            
            return {
                "metadata": sam_metadata,
                "count": len(sam_metadata),
                "image_size": (img_w, img_h),
                "masks": masks
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None