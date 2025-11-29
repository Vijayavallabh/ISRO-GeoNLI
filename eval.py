import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Union
from shapely.geometry import Polygon
import json
import wandb
from dotenv import load_dotenv
load_dotenv()

class GeoNLIEvaluator:
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 spatial_resolution_m: float = None,
                 image_width: int = None,
                 image_height: int = None):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
        self.spatial_resolution_m = spatial_resolution_m
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.weights = {
            'captioning': 0.20,
            'grounding': 0.30,
            'binary': 0.10,
            'numeric': 0.20,
            'semantic': 0.20
        }

    def log_to_wandb(self, scores: Dict[str, float], metadata: Dict[str, any] = None, 
                     predictions: Dict[str, Union[str, float, List]] = None, 
                     ground_truths: Dict[str, Union[str, float, List]] = None,
                     project_name: str = "GeoNLI-Evaluation", run_name: str = None):

        if wandb.run is None:
            wandb.init(project=project_name, name=run_name)
        
        wandb.log(scores)
        
        if metadata:
            wandb.config.update(metadata)
        if predictions:
            wandb.log({"predictions": predictions})
        if ground_truths:
            wandb.log({"ground_truths": ground_truths})   

    def get_bert_embedding(self, text: str) -> torch.Tensor:

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze()
    
    def pixels_to_meters(self, pixel_value: float) -> float:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert pixels to meters.")
        return pixel_value * self.spatial_resolution_m
    
    def meters_to_pixels(self, meter_value: float) -> float:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert meters to pixels.")
        return meter_value / self.spatial_resolution_m
    
    def pixel_area_to_square_meters(self, pixel_area: float) -> float:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert area.")
        return pixel_area * (self.spatial_resolution_m ** 2)
    
    def square_meters_to_pixel_area(self, area_m2: float) -> float:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert area.")
        return area_m2 / (self.spatial_resolution_m ** 2)
    
    def obb_pixel_to_meters(self, obb_pixels: List[float]) -> List[float]:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert OBB.")
        
        cx_m = self.pixels_to_meters(obb_pixels[0])
        cy_m = self.pixels_to_meters(obb_pixels[1])
        width_m = self.pixels_to_meters(obb_pixels[2])
        height_m = self.pixels_to_meters(obb_pixels[3])
        angle = obb_pixels[4] 
        
        return [cx_m, cy_m, width_m, height_m, angle]
    
    def obb_meters_to_pixels(self, obb_meters: List[float]) -> List[float]:

        if self.spatial_resolution_m is None:
            raise ValueError("spatial_resolution_m not set. Cannot convert OBB.")
        
        cx_px = self.meters_to_pixels(obb_meters[0])
        cy_px = self.meters_to_pixels(obb_meters[1])
        width_px = self.meters_to_pixels(obb_meters[2])
        height_px = self.meters_to_pixels(obb_meters[3])
        angle = obb_meters[4]
        
        return [cx_px, cy_px, width_px, height_px, angle]
    
    def normalize_coordinates(self, value: float, dimension: float) -> float:

        if dimension is None or dimension == 0:
            raise ValueError("Image dimension not set or invalid.")
        return value / dimension
    
    def denormalize_coordinates(self, value: float, dimension: float) -> float:

        if dimension is None or dimension == 0:
            raise ValueError("Image dimension not set or invalid.")
        return value * dimension
    
    def obb_normalized_to_absolute(self, obb_norm: List[float]) -> List[float]:

        if self.image_width is None or self.image_height is None:
            raise ValueError("Image dimensions not set. Cannot convert normalized coordinates.")
        
        cx_px = self.denormalize_coordinates(obb_norm[0], self.image_width)
        cy_px = self.denormalize_coordinates(obb_norm[1], self.image_height)
        width_px = self.denormalize_coordinates(obb_norm[2], self.image_width)
        height_px = self.denormalize_coordinates(obb_norm[3], self.image_height)
        angle = obb_norm[4]  
        
        return [cx_px, cy_px, width_px, height_px, angle]
    
    def obb_absolute_to_normalized(self, obb_abs: List[float]) -> List[float]:

        if self.image_width is None or self.image_height is None:
            raise ValueError("Image dimensions not set. Cannot normalize coordinates.")
        
        cx_norm = self.normalize_coordinates(obb_abs[0], self.image_width)
        cy_norm = self.normalize_coordinates(obb_abs[1], self.image_height)
        width_norm = self.normalize_coordinates(obb_abs[2], self.image_width)
        height_norm = self.normalize_coordinates(obb_abs[3], self.image_height)
        angle = obb_abs[4]  
        
        return [cx_norm, cy_norm, width_norm, height_norm, angle]
    
    def validate_obb_bounds(self, obb: List[float], normalized: bool = False) -> bool:

        if self.image_width is None or self.image_height is None:
            return True
        
        cx, cy, width, height, angle = obb
        
        if normalized:
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 
                    0 <= width <= 1 and 0 <= height <= 1):
                return False
        else:
            polygon = self.obb_to_polygon(obb)
            for vertex in polygon:
                if not (0 <= vertex[0] <= self.image_width and 
                        0 <= vertex[1] <= self.image_height):
                    return False
        
        return True
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:

        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:

        if n > len(tokens):
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def bert_bleu_score(self,candidate: str,reference: str,N: int = 4,alpha: float = 0.5,mode: str = "caption",epsilon: float = 1e-8,) -> float:
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()

        Lc = len(candidate_tokens)
        Lr = len(reference_tokens)

        if Lr == 0:
            # No reference content; define score as 0
            return 0.0

        Pn_list = []

        for n in range(1, N + 1):
            Cn = self.get_ngrams(candidate_tokens, n)
            Rn = self.get_ngrams(reference_tokens, n)

            if len(Rn) == 0 or len(Cn) == 0:
                Pn_list.append(0.0)
                continue

            # Precompute embeddings for candidate n-grams
            c_emb_cache = {}
            for c_ngram in Cn:
                if c_ngram not in c_emb_cache:
                    c_emb_cache[c_ngram] = self.get_bert_embedding(c_ngram)

            # Semantic recall over reference n-grams
            sims = []
            for r_ngram in Rn:
                r_emb = self.get_bert_embedding(r_ngram)
                max_sim = 0.0
                for c_ngram, c_emb in c_emb_cache.items():
                    sim = self.cosine_similarity(c_emb, r_emb)
                    if sim > max_sim:
                        max_sim = sim
                sims.append(max_sim)

            Pn = float(np.mean(sims)) if sims else 0.0
            Pn_list.append(Pn)

        Pmax = max(Pn_list) if Pn_list else 0.0

        # Length penalty
        if Lr == 0:
            LP = 1.0
        else:
            length_diff = abs(Lc - Lr) / max(Lr, 1)
            if mode == "caption":
                # LP = exp(-alpha * |Lc - Lr| / Lr)
                LP = float(np.exp(-alpha * length_diff))
            elif mode == "semantic":
                # LP = exp(alpha * (1 - |Lc - Lr| / Lr))
                LP = float(np.exp(alpha * (1.0 - length_diff)))
            else:
                # Fallback: no length penalty
                LP = 1.0

        score = LP * Pmax
        return float(np.clip(score, 0.0, 1.0))

    
    def evaluate_captioning(self, candidate: str, reference: str) -> float:
        return self.bert_bleu_score(candidate, reference, N=4, alpha=0.5, mode="caption")
    
    def polygon_area(self, vertices: np.ndarray) -> float:

        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def polygon_intersection(self, poly1: np.ndarray, poly2: np.ndarray) -> float:

        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            
            if not p1.is_valid:
                p1 = p1.buffer(0)
            if not p2.is_valid:
                p2 = p2.buffer(0)
            
            intersection = p1.intersection(p2)
            return intersection.area
        except:
            return 0.0

    def obb_to_polygon(self, obb: List[float], angle_in_degrees: bool = True) -> np.ndarray:

        cx, cy, w, h, angle = obb
        
        if angle_in_degrees:
            angle = np.radians(angle)
        
        w_half = w / 2
        h_half = h / 2
        
        corners = np.array([
            [-w_half, -h_half],
            [w_half, -h_half],
            [w_half, h_half],
            [-w_half, h_half]
        ])
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        
        return rotated_corners
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:

        poly1 = self.obb_to_polygon(box1)
        poly2 = self.obb_to_polygon(box2)

        area1 = self.polygon_area(poly1)
        area2 = self.polygon_area(poly2)
        
        if area1 == 0 or area2 == 0:
            return 0.0

        intersection_area = self.polygon_intersection(poly1, poly2)

        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def evaluate_grounding(self,pred_boxes: List[List[float]],gt_boxes: List[List[float]],alpha: float = 2.5,coordinate_system: str = 'normalized',validate_bounds: bool = False) -> float:
        if coordinate_system == 'meter':
            if self.spatial_resolution_m is None:
                raise ValueError("spatial_resolution_m required for meter coordinates")
            pred_boxes = [self.obb_meters_to_pixels(box) for box in pred_boxes]
            gt_boxes = [self.obb_meters_to_pixels(box) for box in gt_boxes]
        elif coordinate_system == 'normalized':
            if self.image_width is None or self.image_height is None:
                raise ValueError("image dimensions required for normalized coordinates")
            pred_boxes = [self.obb_normalized_to_absolute(box) for box in pred_boxes]
            gt_boxes = [self.obb_normalized_to_absolute(box) for box in gt_boxes]

        if validate_bounds:
            for i, box in enumerate(pred_boxes):
                if not self.validate_obb_bounds(box, normalized=False):
                    print(f"Warning: Predicted box {i} is out of bounds")
            for i, box in enumerate(gt_boxes):
                if not self.validate_obb_bounds(box, normalized=False):
                    print(f"Warning: Ground truth box {i} is out of bounds")

        N_pred = len(pred_boxes)
        N_ref = len(gt_boxes)

        count_penalty = float(np.exp(-alpha * abs(N_pred - N_ref)))

        if N_pred == 0 or N_ref == 0:
            return count_penalty * 0.0

        ious = []
        used_gt = set()

        for pred_box in pred_boxes:
            max_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                iou = self.compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                used_gt.add(best_gt_idx)
                ious.append(max_iou)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        grounding_score = count_penalty * mean_iou

        return float(np.clip(grounding_score, 0.0, 1.0))

    
    def evaluate_binary(self, prediction: str, ground_truth: str) -> float:

        pred_normalized = prediction.strip().lower()
        gt_normalized = ground_truth.strip().lower()
        
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    def evaluate_numeric(self,prediction: float,ground_truth: float,unit: str = None,prediction_unit: str = None,alpha: float = 23.0,) -> float:
        # Unit conversion (unchanged)
        if unit and prediction_unit and unit != prediction_unit:
            if self.spatial_resolution_m is None:
                raise ValueError("spatial_resolution_m required for unit conversion")

            if prediction_unit == 'pixels' and unit == 'meters':
                prediction = self.pixels_to_meters(prediction)
            elif prediction_unit == 'meters' and unit == 'pixels':
                prediction = self.meters_to_pixels(prediction)
            elif prediction_unit == 'square_pixels' and unit == 'square_meters':
                prediction = self.pixel_area_to_square_meters(prediction)
            elif prediction_unit == 'square_meters' and unit == 'square_pixels':
                prediction = self.square_meters_to_pixel_area(prediction)
            else:
                raise ValueError(f"Unsupported unit conversion: {prediction_unit} to {unit}")

        # Relative error–based exponential decay
        if ground_truth == 0:
            # Degenerate case: fall back to absolute error
            err = abs(prediction - ground_truth)
        else:
            err = abs(prediction - ground_truth) / abs(ground_truth)

        score = float(np.exp(-alpha * err))
        return float(np.clip(score, 0.0, 1.0))

    
    def evaluate_semantic(self, prediction: str, ground_truth: str) -> float:
        return self.bert_bleu_score(prediction,ground_truth,N=4,alpha=0.5,mode="semantic",)

    
    def compute_final_score(self, scores: Dict[str, float]) -> float:

        final_score = 0.0
        
        for task, weight in self.weights.items():
            if task in scores:
                final_score += weight * scores[task]
        
        return float(np.clip(final_score, 0, 1))
    
    def parse_response_json(self, response_json: dict) -> (Dict[str, Union[str, float, List]], Dict[str, any]):

        metadata = response_json.get('input_image', {}).get('metadata', {})
        
        self.image_width = metadata.get('width')
        self.image_height = metadata.get('height')
        self.spatial_resolution_m = metadata.get('spatial_resolution_m')
        
        predictions = {}
        
        queries = response_json.get('queries', {})
        
        if 'caption_query' in queries:
            predictions['captioning'] = queries['caption_query'].get('response', '')
        
        if 'grounding_query' in queries:
            grounding_response = queries['grounding_query'].get('response', [])
            predictions['grounding'] = [item.get('obbox', []) for item in grounding_response]
            
        
        attr = queries.get('attribute_query', {})
        
        if 'binary' in attr:
            predictions['binary'] = attr['binary'].get('response', '')
        
        if 'numeric' in attr:
            predictions['numeric'] = attr['numeric'].get('response', 0.0)

        if 'semantic' in attr:
            predictions['semantic'] = attr['semantic'].get('response', '')
        
        return predictions, metadata
    
    def load_predictions_from_json(self, json_path: str) -> (Dict[str, Union[str, float, List]], Dict[str, any]):
   
        with open(json_path, 'r') as f:
            data = json.load(f)
        return self.parse_response_json(data)
    
    def evaluate_all(self, predictions: Dict[str, Union[str, float, List]], 
                     ground_truths: Dict[str, Union[str, float, List]],
                     metadata: Dict[str, any] = None) -> Dict[str, float]:

        scores = {}
        metadata = metadata or {}
        
        if 'captioning' in predictions and 'captioning' in ground_truths:
            scores['captioning'] = self.evaluate_captioning(
                predictions['captioning'], ground_truths['captioning']
            )
        
        if 'grounding' in predictions and 'grounding' in ground_truths:
            scores['grounding'] = self.evaluate_grounding(
                predictions['grounding'], ground_truths['grounding'],
                coordinate_system='normalized', validate_bounds=True
            )
        
        if 'binary' in predictions and 'binary' in ground_truths:
            scores['binary'] = self.evaluate_binary(
                predictions['binary'], ground_truths['binary']
            )
        
        if 'numeric' in predictions and 'numeric' in ground_truths:
            scores['numeric'] = self.evaluate_numeric(
                predictions['numeric'], ground_truths['numeric'],
                unit=None, prediction_unit=None
            )
        
        if 'semantic' in predictions and 'semantic' in ground_truths:
            scores['semantic'] = self.evaluate_semantic(
                predictions['semantic'], ground_truths['semantic']
            )
        
        scores['final_score'] = self.compute_final_score(scores)
        
        return scores

def print_scores(scores, weights):
    """Helper function to print scores in a formatted way."""
    print("\nResults:")
    for task, score in scores.items():
        if task != 'final_score':
            weight = weights.get(task, 0)
            print(f"  {task.capitalize():15s}: {score:.4f} (weight: {weight:.0%})")
    print(f"  {'FINAL SCORE':15s}: {scores['final_score']:.4f}")

# Example usage
if __name__ == "__main__":
    print("Initializing GeoNLI Evaluator ...")
    print("="*60)
    evaluator = GeoNLIEvaluator()
    predictions, metadata = evaluator.load_predictions_from_json('sample_dataset_inter_iit_v1_2/sample2_response.json')
    
    print(f"Spatial Resolution: {evaluator.spatial_resolution_m} meters/pixel")
    print(f"Image Dimensions: {evaluator.image_width} x {evaluator.image_height} pixels")
    print("OBB Format: [center_x, center_y, width, height, angle]")
    print("="*60)
    
    ground_truths = predictions.copy()
    
    scores = evaluator.evaluate_all(predictions, ground_truths, metadata)
    print_scores(scores, evaluator.weights)
    evaluator.log_to_wandb(scores, metadata, predictions, ground_truths, 
                           project_name="GeoNLI-Evaluation", run_name="sample2_run")
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)



