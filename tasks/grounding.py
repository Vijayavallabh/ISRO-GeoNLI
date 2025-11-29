"""
Object grounding task implementation.
"""

import re
import matplotlib.pyplot as plt
from utils.visualization import annotate_image_with_boxes


class GroundingTask:
    """Handles object detection and grounding."""
    
    def __init__(self, vlm_interface, sam3_interface):
        """
        Args:
            vlm_interface: VLMInterface instance
            sam3_interface: SAM3Interface instance
        """
        self.vlm = vlm_interface
        self.sam3 = sam3_interface
    
    def extract_target_classes(self, image, query):
        """
        Extract target object categories from query.
        
        Args:
            image: PIL Image
            query: User query string
            
        Returns:
            List of target class names
        """
        prompt = (
            f"Analyze the user query: '{query}'.\n"
            "Identify ALL distinct object categories that need to be detected.\n"
            "Return a comma-separated list of category names (singular form). "
            "Do not add punctuation or filler text.\n"
            "Example 1: 'Find the red cars and the pool' -> car, swimming pool\n"
            "Example 2: 'Locate the planes' -> plane\n"
            "Target Categories:"
        )
        
        response = self.vlm.query(image, prompt, max_tokens=32)
        clean_resp = re.sub(r"[\[\]\'\"<>]", "", response)
        classes = [c.strip().lower() for c in clean_resp.split(',') if c.strip()]
        
        print(f"   [Extraction] Query: '{query}' -> Targets: {classes}")
        return classes
    
    def detect_coarse_boxes(self, image, description, norm_scale=1000):
        """
        Detect coarse bounding boxes using VLM.
        
        Args:
            image: PIL Image
            description: Text description of targets
            norm_scale: Coordinate normalization scale
            
        Returns:
            List of absolute coordinate boxes [[x1,y1,x2,y2], ...]
        """
        prompt = (
            f"Detect the object described by: '{description}'.\n"
            f"Return the bounding boxes in [x1, y1, x2, y2] format.\n"
            f"Use a coordinate scale of 0-{norm_scale}.\n"
            "Return ONLY the Python list of lists. Example: [[100, 200, 500, 600]]"
        )
        
        response = self.vlm.query(image, prompt, max_tokens=128)
        
        # Parse response
        try:
            match = re.search(r"\[\[.*?\]\]", response)
            if match:
                hbb_list_norm = eval(match.group(0))
            else:
                match_single = re.search(r"\[\d+.*?\]", response)
                if match_single:
                    hbb_list_norm = [eval(match_single.group(0))]
                else:
                    return []
        except:
            print(f"   [Warning] Parsing failed for coarse detection.")
            return []
        
        # Convert to absolute pixel coordinates
        w, h = image.size
        abs_hbbs = []
        for box in hbb_list_norm:
            if len(box) == 4:
                x1 = box[0] * w / norm_scale
                y1 = box[1] * h / norm_scale
                x2 = box[2] * w / norm_scale
                y2 = box[3] * h / norm_scale
                abs_hbbs.append([x1, y1, x2, y2])
        
        return abs_hbbs
    
    def ground_objects(self, image, query, gsd=1.0, score_threshold=0.4, 
                      forced_targets=None, show_visualization=True):
        """
        Perform complete grounding pipeline.
        
        Args:
            image: PIL Image
            query: User query
            gsd: Ground Sample Distance
            score_threshold: Confidence threshold
            forced_targets: Optional pre-specified target classes
            show_visualization: Whether to display results
            
        Returns:
            List of detection dictionaries
        """
        if forced_targets:
            target_classes = forced_targets
            description = query if query else f"Find {', '.join(forced_targets)}"
            print(f"--- Task: Grounding (Dynamic for: {target_classes}) ---")
        else:
            print(f"--- Task: Grounding (Query: '{query}') ---")
            target_classes = self.extract_target_classes(image, query)
            description = query
        
        all_detections = []
        global_id = 1
        
        # Stage 1: Coarse detection with VLM
        print(f"   -> Step 1: Detecting coarse boxes...")
        coarse_hbbs = self.detect_coarse_boxes(image, description)
        
        if not coarse_hbbs:
            print("      No coarse boxes found.")
            return []
        
        print(f"      Found {len(coarse_hbbs)} coarse box(es). Refining...")
        
        # Stage 2: Refinement with SAM 3
        for target in target_classes:
            print(f"   -> Step 2: Refining for target class '{target}'...")
            
            for hbb in coarse_hbbs:
                result = self.sam3.refine_detection(
                    image, hbb, target, 
                    score_threshold=score_threshold, 
                    gsd=gsd
                )
                
                if result and result['score'] > score_threshold:
                    # De-duplication
                    if not self._is_duplicate(result, all_detections):
                        result['id'] = global_id
                        all_detections.append(result)
                        global_id += 1
        
        print(f"   -> Found {len(all_detections)} verified objects.")
        
        # Visualization
        if show_visualization and all_detections:
            final_obbs = [d['obb'] for d in all_detections]
            annotated_preview, _ = annotate_image_with_boxes(image, final_obbs)
            
            plt.figure(figsize=(12, 12))
            plt.imshow(annotated_preview)
            plt.axis('off')
            plt.title(f"Grounding Result: {description}")
            plt.show()
        
        return all_detections
    
    def _is_duplicate(self, new_det, existing_dets, distance_threshold=20):
        """Check if detection is duplicate based on center point distance."""
        new_center = new_det['center_point']
        
        for existing in existing_dets:
            ex_center = existing['center_point']
            dist = ((ex_center[0] - new_center[0])**2 + 
                   (ex_center[1] - new_center[1])**2)**0.5
            
            if dist < distance_threshold:
                return True
        
        return False
