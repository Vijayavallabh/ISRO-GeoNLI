import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Union
#from shapely.geometry import Polygon
import json
import wandb
from collections import defaultdict
import jsonlines
import os
from tqdm import tqdm
import re

WORD_TO_NUM = {
        'no': 0, 'none': 0, 'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

WEIGHTS = {
        'captioning': 0.20,
        'grounding': 0.30,
        'binary': 0.10,
        'numeric': 0.20,
        'semantic': 0.20
        }

# --- Evaluator Class ---

class GeoNLIEvaluator:

    def __init__(self, model_name: str = 'distilbert-base-uncased',
            spatial_resolution_m: float = None,
            image_width: int = None,
            image_height: int = None,
            batch_size: int = 32):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing GeoNLIEvaluator on {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.spatial_resolution_m = spatial_resolution_m
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.weights = {
                'captioning': 0.20,
                'grounding': 0.30,
                'binary': 0.10,
                'numeric': 0.20,
                'semantic': 0.20
                }
      # --- Unit Conversion Helpers ---
    def pixels_to_meters(self, pixel_value: float) -> float:
        if self.spatial_resolution_m is None: raise ValueError("spatial_resolution_m not set.")
        return pixel_value * self.spatial_resolution_m

    def meters_to_pixels(self, meter_value: float) -> float:
        if self.spatial_resolution_m is None: raise ValueError("spatial_resolution_m not set.")
        return meter_value / self.spatial_resolution_m

    def pixel_area_to_square_meters(self, pixel_area: float) -> float:
        if self.spatial_resolution_m is None: raise ValueError("spatial_resolution_m not set.")
        return pixel_area * (self.spatial_resolution_m ** 2)

    def square_meters_to_pixel_area(self, area_m2: float) -> float:
        if self.spatial_resolution_m is None: raise ValueError("spatial_resolution_m not set.")
        return area_m2 / (self.spatial_resolution_m ** 2)

    def obb_pixel_to_meters(self, obb_pixels: List[float]) -> List[float]:
        cx_m = self.pixels_to_meters(obb_pixels[0])
        cy_m = self.pixels_to_meters(obb_pixels[1])
        width_m = self.pixels_to_meters(obb_pixels[2])
        height_m = self.pixels_to_meters(obb_pixels[3])
        angle = obb_pixels[4]
        return [cx_m, cy_m, width_m, height_m, angle]

    def obb_meters_to_pixels(self, obb_meters: List[float]) -> List[float]:
        cx_px = self.meters_to_pixels(obb_meters[0])
        cy_px = self.meters_to_pixels(obb_meters[1])
        width_px = self.meters_to_pixels(obb_meters[2])
        height_px = self.meters_to_pixels(obb_meters[3])
        angle = obb_meters[4]
        return [cx_px, cy_px, width_px, height_px, angle]

    # --- Coordinate Normalization Helpers ---
    def normalize_coordinates(self, value: float, dimension: float) -> float:
        if dimension is None or dimension == 0: raise ValueError("Image dimension invalid.")
        return value / dimension

    def denormalize_coordinates(self, value: float, dimension: float) -> float:
        if dimension is None or dimension == 0: raise ValueError("Image dimension invalid.")
        return value * dimension

    def obb_normalized_to_absolute(self, obb_norm: List[float]) -> List[float]:
        if self.image_width is None or self.image_height is None: raise ValueError("Image dims not set.")
        cx_px = self.denormalize_coordinates(obb_norm[0], self.image_width)
        cy_px = self.denormalize_coordinates(obb_norm[1], self.image_height)
        width_px = self.denormalize_coordinates(obb_norm[2], self.image_width)
        height_px = self.denormalize_coordinates(obb_norm[3], self.image_height)
        return [cx_px, cy_px, width_px, height_px, obb_norm[4]]

    def validate_obb_bounds(self, obb: List[float], normalized: bool = False) -> bool:
        if self.image_width is None or self.image_height is None: return True
        # Simplified validation logic
        return True

    def get_bert_embedding_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Computes embeddings for a list of texts in batches to maximize GPU usage.
        """
        if not texts:
            return torch.tensor([]).to(self.device)

        all_embeddings = []

        # Process in chunks of self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # CLS token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(cls_embeddings)

        return torch.cat(all_embeddings, dim=0)

    # --- NLP Scoring ---
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if n > len(tokens): return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def preprocess_text_to_ngrams(self, texts: List[str], N: int = 4) -> List[Dict[int, List[str]]]:
        """
            Precomputes n-grams for a list of texts. 
            Returns a list of dictionaries, where each dict maps 'n' to a list of n-gram strings.
            """
        processed_data = []
        # Pre-tokenize all texts (CPU intensive)
        for text in texts:
            tokens = text.lower().split()
            ngram_dict = {}
            for n in range(1, N + 1):
                ngram_dict[n] = self.get_ngrams(tokens, n)
            processed_data.append(ngram_dict)
        return processed_data

    def batch_bert_bleu_score(self, 
            cand_ngrams_batch: List[Dict[int, List[str]]], 
            ref_ngrams_batch: List[Dict[int, List[str]]], 
            N: int = 4, 
            epsilon: float = 1e-8) -> List[float]:

        if not cand_ngrams_batch: 
            return []

        batch_size = len(cand_ngrams_batch)
        scores = [[] for _ in range(batch_size)]

        c_lens = []
        r_lens = []
        for i in range(batch_size):
            c_tokens = cand_ngrams_batch[i].get(1, [])
            r_tokens = ref_ngrams_batch[i].get(1, [])
            c_lens.append(len(c_tokens))
            r_lens.append(len(r_tokens))
        # Collect ALL unique n-grams needed for this specific batch
        # This ensures we only run the BERT model on unique phrases
        all_unique_ngrams = set()

        # We need to track which n-grams belong to which sample to reconstruct scores later
        # Structure: [(sample_idx, n, c_ngrams, r_ngrams), ...]
        batch_metadata = []

        for n in range(1, N + 1):
            for i in range(batch_size):
                c_ngrams = cand_ngrams_batch[i].get(n, [])
                r_ngrams = ref_ngrams_batch[i].get(n, [])

                if not c_ngrams or not r_ngrams:
                    batch_metadata.append({
                        'sample_idx': i, 'empty': True
                        })
                else:
                    batch_metadata.append({
                        'sample_idx': i,
                        'c_ngrams': c_ngrams,
                        'r_ngrams': r_ngrams,
                        'empty': False
                        })
                    all_unique_ngrams.update(c_ngrams)
                    all_unique_ngrams.update(r_ngrams)

        # --- GPU STEP: Batch Embeddings ---
        unique_ngrams_list = list(all_unique_ngrams)
        ngram_to_emb = {}

        if unique_ngrams_list:
            unique_embs = self.get_bert_embedding_batch(unique_ngrams_list)
            # Normalizing ensures cosine similarity is just a dot product later
            unique_embs = F.normalize(unique_embs, p=2, dim=1)

            # Map string -> embedding vector
            for idx, ngram in enumerate(unique_ngrams_list):
                ngram_to_emb[ngram] = unique_embs[idx]

        # --- SCORING STEP ---
        for data in batch_metadata:
            i = data['sample_idx']

            if data['empty']:
                continue

            c_ngrams = data['c_ngrams']
            r_ngrams = data['r_ngrams']

            # Retrieve embeddings from the dictionary (Fast lookup)
            C_emb = torch.stack([ngram_to_emb[ng] for ng in c_ngrams])
            R_emb = torch.stack([ngram_to_emb[ng] for ng in r_ngrams])

            # Matrix Multiplication for Cosine Similarity
            sim_matrix = torch.matmul(C_emb, R_emb.T)
            max_similarities_per_ref, _ = torch.max(sim_matrix, dim=0)
            total_similarity = torch.sum(max_similarities_per_ref).item()

            recall_n = total_similarity / len(r_ngrams)
            scores[i].append(recall_n)

        # Aggregate scores
        final_scores = []
        for score_list, c_len, r_len in zip(scores, c_lens, r_lens):

            # Handle edge case: empty reference (prevent division by zero)
            if r_len == 0:
                final_scores.append(0.0)
                continue

            # PDF Eq 140: Length Penalty
            # LP = exp(-0.5 * |L_C - L_R| / L_R)
            lp_score = np.exp(-0.5 * abs(c_len - r_len) / r_len)

            # Get valid P_n scores
            valid_scores = [s for s in score_list if s > 0]

            if not valid_scores:
                final_scores.append(0.0)
            else:
                # PDF Eq 138: S = LP * Max(Pn)
                # We use MAX, not MEAN 
                max_p = max(valid_scores)
                final_score = lp_score * max_p
                final_scores.append(float(np.clip(final_score, 0, 1)))

        return final_scores

    def polygon_area(self, vertices: np.ndarray) -> float:
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def polygon_intersection(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            if not p1.is_valid: p1 = p1.buffer(0)
            if not p2.is_valid: p2 = p2.buffer(0)
            intersection = p1.intersection(p2)
            return intersection.area
        except:
            return 0.0

    def obb_to_polygon(self, obb: List[float], angle_in_degrees: bool = True) -> np.ndarray:
        cx, cy, w, h, angle = obb
        if angle_in_degrees: angle = np.radians(angle)
        w_half, h_half = w / 2, h / 2
        corners = np.array([[-w_half, -h_half], [w_half, -h_half], [w_half, h_half], [-w_half, h_half]])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        return rotated_corners

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        poly1 = self.obb_to_polygon(box1)
        poly2 = self.obb_to_polygon(box2)
        area1 = self.polygon_area(poly1)
        area2 = self.polygon_area(poly2)
        if area1 == 0 or area2 == 0: return 0.0
        intersection_area = self.polygon_intersection(poly1, poly2)
        union_area = area1 + area2 - intersection_area
        if union_area == 0: return 0.0
        return intersection_area / union_area

    def evaluate_grounding(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]], alpha: float = 1.0, coordinate_system: str = 'normalized', validate_bounds: bool = False) -> float:
        if coordinate_system == 'normalized':
            pred_boxes = [self.obb_normalized_to_absolute(box) for box in pred_boxes]
            gt_boxes = [self.obb_normalized_to_absolute(box) for box in gt_boxes]

        N_pred, N_ref = len(pred_boxes), len(gt_boxes)
        count_penalty = np.exp(-alpha * abs(N_pred - N_ref))
        if N_pred == 0 or N_ref == 0: return 0.0

        ious = []
        used_gt = set()
        for pred_box in pred_boxes:
            max_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt: continue
                iou = self.compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0:
                used_gt.add(best_gt_idx)
                ious.append(max_iou)

        mean_iou = np.mean(ious) if ious else 0.0
        return float(np.clip(count_penalty * mean_iou, 0, 1))

    def evaluate_numeric(self, prediction: float, ground_truth: float) -> float:
        if ground_truth == 0:
            # Avoid division by zero; if both are 0, perfect match.
            return 1.0 if prediction == 0 else 0.0

        # PDF Formula: exp(-alpha * |pred - gt| / gt) with alpha = 23
        alpha = 23.0
        relative_error = abs(prediction - ground_truth) / ground_truth
        score = np.exp(-alpha * relative_error)
        return float(np.clip(score, 0, 1))

    def evaluate_binary(self, pred: str, gt: str) -> float:
        """Exact Match {0, 1}"""
        p = str(pred).strip().lower()
        g = str(gt).strip().lower()
        if 'yes' in p and 'yes' in g: return 1.0
        if 'no' in p and 'no' in g: return 1.0
        return 1.0 if p == g else 0.0

    def compute_final_score(self, scores: Dict[str, float]) -> float:
        final_score = 0.0
        for task, weight in WEIGHTS.items():
            if task in scores: final_score += weight * scores[task]
        return float(np.clip(final_score, 0, 1))

def convert_corners_to_obb(corners: list) -> list:
    """Converts 8-point corner coordinates to [cx, cy, w, h, angle]."""
    if len(corners) != 8:
        if len(corners) == 4: # xyxy fallback
            w = corners[2] - corners[0]
            h = corners[3] - corners[1]
            cx = corners[0] + w/2
            cy = corners[1] + h/2
            return [cx, cy, w, h, 0.0]
        return corners

    pts = np.array(corners).reshape(-1, 2)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    w = np.linalg.norm(pts[0] - pts[1])
    h = np.linalg.norm(pts[1] - pts[2])
    dx = pts[1, 0] - pts[0, 0]
    dy = pts[1, 1] - pts[0, 1]
    angle_deg = np.degrees(np.arctan2(dy, dx))

    return [float(cx), float(cy), float(w), float(h), float(angle_deg)]

def clean_id(raw_id):
    """Normalize ID: remove extension and folder path"""
    s = str(raw_id)
    s = os.path.basename(s)
    s = s.replace(".png", "").replace(".jpg", "").replace(".json", "")
    return s.strip()

def parse_numeric_value(val):
    """Parses a value into a float, checking WORD_TO_NUM first."""
    s_val = str(val).strip().lower()

    # 1. Check text mapping
    if s_val in WORD_TO_NUM:
        return float(WORD_TO_NUM[s_val])

    # 2. Try standard float conversion
    try:
        return float(val)
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoNLI Model Predictions")

    parser.add_argument('--pred', type=str, default=None, help="Path to model predictions .jsonl")
    parser.add_argument('--gt', type=str, default=None, help="Path to Ground Truth .jsonl")
    parser.add_argument('--img_width', type=int, default=512, help="Image width for normalization")
    parser.add_argument('--img_height', type=int, default=512, help="Image height for normalization")

    # Pass an empty list to parse_args() to prevent it from trying to parse kernel arguments
    args = parser.parse_args()

    # instantiate the evaluator
    evaluator = GeoNLIEvaluator(
            model_name='distilbert-base-uncased',
            image_width=512,
            image_height=512
            )

    # load the predictions_jsonl into a list
    predictions_list = []
    with jsonlines.open(args.pred) as reader:
        for obj in reader:
            predictions_list.append(obj)

    # loads the ground_truth_jsonl into a list
    gt_list = []
    with jsonlines.open(args.gt) as reader:
        for obj in reader:
            gt_list.append(obj)

    # convert gt_list to gt_map:
    gt_map = {}
    for obj in gt_list:
        image = clean_id(obj['image'])
        gt_map[image] = {}
        gt_map[image]['caption'] = obj['caption']
        gt_map[image]['vqa'] = obj['qa_pairs']
        gt_map[image]['grounding'] = None

    print(f"Loaded predictions. Total predictions count: {len(predictions_list)}")
    print(f"Loaded ground truth. Total ground truth count: {len(gt_list)}")

    scores = {'Grounding': [], 'Binary': [], 'Numeric': []}
    batch_data = {
            'captioning': {'cands': [], 'refs': []},
            'semantic': {'cands': [], 'refs': []}
            }
    for pred_obj in tqdm(predictions_list, desc = 'Processing Metadata'):
        img_id = clean_id(pred_obj.get('image', ''))
        gt = gt_map.get(img_id)

        if not gt: continue

        if gt['caption']:
            batch_data['captioning']['cands'].append(pred_obj.get('caption', ''))
            batch_data['captioning']['refs'].append(gt['caption'])

        if gt['vqa']:
            p_qa_list = pred_obj.get('qa_pairs', [])
            p_map = {}
            for p_item in p_qa_list:
                q_text = p_item.get('question', '')
            clean_q = re.sub(r'\W+', '', q_text.lower())
            p_map[clean_q] = str(p_item.get('answer', ''))

            # Match with GT
            for gq in gt['vqa']:
                q_text_gt = gq.get('question', '')
                clean_g = re.sub(r'\W+', '', q_text_gt.lower())

                if clean_g in p_map:
                    p_ans = p_map[clean_g]
                    m = re.search(r"<answer>(.*?)</answer>", p_ans, flags=re.DOTALL)
                    p_ans = m.group(1).strip() if m else p_ans
                    g_ans = str(gq.get('answer', ''))
                    q_type = gq.get('type', 'semantic')
                    g_ans_clean = g_ans.strip().lower()

                    # Fast Evaluation (Binary/Numeric)
                    if ('existence' in q_type) or ('yes' in g_ans.lower() or 'no' in g_ans.lower()):
                        scores["Binary"].append(evaluator.evaluate_binary(p_ans, g_ans))
                    elif ('quantity' in q_type) or (g_ans.isdigit() or g_ans_clean in WORD_TO_NUM):
                        p_val = parse_numeric_value(p_ans)
                        g_val = parse_numeric_value(g_ans)
                        scores["Numeric"].append(evaluator.evaluate_numeric(p_val, g_val))
                    else:
                        # Collect for Batch Evaluation (Semantic)
                        batch_data["semantic"]["cands"].append(p_ans)
                        batch_data["semantic"]["refs"].append(g_ans)

        # --- C. Process Grounding ---
        if gt['grounding']:
            p_objects = pred_obj.get('objects', [])
            p_boxes = []
            for obj in p_objects:
                if isinstance(obj, dict): p_boxes.append(obj.get('obj_corner', []))
                elif isinstance(obj, list): p_boxes.append(obj)

            g_boxes = []
            for g_item in gt['grounding']:
                if 'obj_corner' in g_item: g_boxes.append(g_item['obj_corner'])

            if g_boxes:
                scores["Grounding"].append(evaluator.evaluate_grounding(p_boxes, g_boxes))

    print('Evaluating BERT based metrics')
    EVAL_BATCH_SIZE = 1024

    # --- 1. Captioning Score ---
    scores["Captioning"] = []
    cap_cands = batch_data["captioning"]["cands"]
    cap_refs = batch_data["captioning"]["refs"]


    if cap_cands:
        print(f" ...Precomputing N-Grams for {len(cap_cands)} captions (CPU)...")
        # Step 1: Precompute everything outside the batch loop
        precomputed_cands = evaluator.preprocess_text_to_ngrams(cap_cands)
        precomputed_refs = evaluator.preprocess_text_to_ngrams(cap_refs)

        print(f" ...Computing Embeddings and Scores (GPU)...")
        # Step 2: Loop only for the GPU memory constrained part
        for i in tqdm(range(0, len(precomputed_cands), EVAL_BATCH_SIZE), desc="Caption Scores"):
            batch_c = precomputed_cands[i : i + EVAL_BATCH_SIZE]
            batch_r = precomputed_refs[i : i + EVAL_BATCH_SIZE]

            # Pass the structured data, not the raw strings
            batch_scores = evaluator.batch_bert_bleu_score(batch_c, batch_r)
            scores["Captioning"].extend(batch_scores)


    # --- 2. Semantic VQA Score ---
    scores["Semantic"] = []

    sem_cands = batch_data["semantic"]["cands"]
    sem_refs = batch_data["semantic"]["refs"]


    if sem_cands:
        print(f" ...Precomputing N-Grams for {len(sem_cands)} semantic VQA answers...")
        precomputed_sem_cands = evaluator.preprocess_text_to_ngrams(sem_cands)
        precomputed_sem_refs = evaluator.preprocess_text_to_ngrams(sem_refs)


        for i in tqdm(range(0, len(precomputed_sem_cands), EVAL_BATCH_SIZE), desc="Semantic Scores"):
            batch_c = precomputed_sem_cands[i : i + EVAL_BATCH_SIZE]
            batch_r = precomputed_sem_refs[i : i + EVAL_BATCH_SIZE]

            batch_scores = evaluator.batch_bert_bleu_score(batch_c, batch_r)
            scores["Semantic"].extend(batch_scores)
    # 4. Final Calculation & Report
    print("\n" + "="*50)
    print("Results")
    print("="*50)

    final_score = 0.0
    for cat, weight in WEIGHTS.items():
        lookup_key = cat.capitalize()
        vals = scores.get(lookup_key, [])
        avg = np.mean(vals) if vals else 0.0000
        weighted_val = avg * weight
        final_score += weighted_val
        print(f"  {lookup_key:<12} {avg:.4f}")

    print("-" * 50)
    print(f"Final Weighted Score: {final_score:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()

