import os
import json
from typing import List, Dict, Union, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch import amp
from transformers import BertTokenizer, BertModel
from shapely.geometry import Polygon
import wandb
from dotenv import load_dotenv

load_dotenv()


class GeoNLIEvaluator:
    """
    GeoNLI Evaluator with GPU-optimized BERT usage and batched n-gram embeddings.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        spatial_resolution_m: float = None,
        image_width: int = None,
        image_height: int = None,
        metric_spec: Dict[str, Any] = None,
        device: str = None,
        use_fp16: bool = True,
    ):
        # --------------------
        # Device / precision
        # --------------------
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        # Enable common inference optimizations
        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Allow TF32 on Ampere+ for faster matmuls if acceptable
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

        # --------------------
        # Language model
        # --------------------
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Simple CPU-side embedding cache: text -> tensor[hidden_size] (L2-normalized)
        self._emb_cache: Dict[str, torch.Tensor] = {}

        # --------------------
        # Image metadata
        # --------------------
        self.spatial_resolution_m = spatial_resolution_m
        self.image_width = image_width
        self.image_height = image_height

        # =======================
        # METRIC SPEC (EDIT HERE)
        # =======================
        default_weights = {
            'captioning': 0.20,
            'grounding': 0.30,
            'binary':    0.10,
            'numeric':   0.20,
            'semantic':  0.20,
        }

        metric_spec = metric_spec or {}
        self.weights: Dict[str, float] = metric_spec.get("weights", default_weights)

        # Captioning metric hyperparameters
        self.captioning_cfg = {
            "N": metric_spec.get("captioning_N", 4),
            "alpha": metric_spec.get("captioning_alpha", 0.5),
            "mode": metric_spec.get("captioning_mode", "caption"),
        }

        # Grounding metric hyperparameters
        self.grounding_cfg = {
            "alpha": metric_spec.get("grounding_alpha", 2.5),
            "coordinate_system": metric_spec.get("grounding_coordinate_system", "normalized"),
            "validate_bounds": metric_spec.get("grounding_validate_bounds", True),
            "iou_threshold": metric_spec.get("grounding_iou_threshold", 0.0),
        }

        # Numeric metric hyperparameters
        self.numeric_cfg = {
            "alpha": metric_spec.get("numeric_alpha", 23.0),
            "default_unit": metric_spec.get("numeric_default_unit", None),
        }

    # =====================
    # Logging
    # =====================

    def log_to_wandb(
        self,
        scores: Dict[str, float],
        metadata: Dict[str, Any] = None,
        predictions: Dict[str, Union[str, float, List]] = None,
        ground_truths: Dict[str, Union[str, float, List]] = None,
        project_name: str = "GeoNLI-Evaluation",
        run_name: str = None,
    ):
        if wandb.run is None:
            wandb.init(project=project_name, name=run_name)

        wandb.log(scores)

        if metadata:
            wandb.config.update(metadata)
        if predictions:
            wandb.log({"predictions": predictions})
        if ground_truths:
            wandb.log({"ground_truths": ground_truths})

    # =====================
    # Text / embedding utils
    # =====================

    def get_bert_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Return L2-normalized CLS embeddings for a list of texts.

        Embeddings are cached on CPU and moved to self.device on demand.
        Uses torch.inference_mode and optional FP16 for fast GPU inference. [web:18][web:28]
        """
        if not texts:
            hidden_size = self.model.config.hidden_size
            return torch.empty(0, hidden_size, device=self.device)

        # Select uncached texts
        uncached = [t for t in texts if t not in self._emb_cache]

        if uncached:
            # Tokenize and move to device
            inputs = self.tokenizer(
                uncached,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=64,  # n-grams are short; keeps inference fast
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference-only forward pass
            with torch.inference_mode():
                if self.use_fp16:
                    with amp.autocast(dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            # CLS embeddings
            cls_embs = outputs.last_hidden_state[:, 0, :].float()
            # L2-normalize to make cosine similarity a dot product
            cls_embs = nn.functional.normalize(cls_embs, p=2, dim=-1)

            # Cache on CPU to save GPU memory
            for text, emb in zip(uncached, cls_embs):
                self._emb_cache[text] = emb.cpu()

        # Stack in input order, move to device
        stacked = torch.stack([self._emb_cache[t] for t in texts], dim=0).to(self.device)
        return stacked

    def get_bert_embedding(self, text: str) -> torch.Tensor:
        """
        Backwards-compatible single-text helper using the batched path.
        """
        return self.get_bert_embeddings_batch([text])[0]

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Cosine similarity between two embeddings (expects 1D tensors).
        If embeddings are normalized, this is equivalent to dot product.
        """
        emb1 = emb1 / (emb1.norm(p=2) + 1e-8)
        emb2 = emb2 / (emb2.norm(p=2) + 1e-8)
        return torch.dot(emb1, emb2).item()

    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if n > len(tokens):
            return []
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # =====================
    # Unit conversions
    # =====================

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

    # =====================
    # OBB conversions
    # =====================

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

    # =====================
    # Captioning metric
    # =====================

    def bert_bleu_score(
        self,
        candidate: str,
        reference: str,
        N: int = 4,
        alpha: float = 0.5,
        mode: str = "caption",
        epsilon: float = 1e-8,
    ) -> float:
        """
        Semantic n-gram recall with length penalty, GPU-optimized.

        All n-grams across orders 1..N are embedded once in batch on GPU,
        then cosine similarities are computed as dot products. [web:2][web:18]
        """
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()

        Lc = len(candidate_tokens)
        Lr = len(reference_tokens)

        if Lr == 0:
            return 0.0

        # Collect n-grams for all orders and build a global set
        all_ngrams_set = set()
        ngram_lists: Dict[int, Tuple[List[str], List[str]]] = {}

        for n in range(1, N + 1):
            Cn = self.get_ngrams(candidate_tokens, n)
            Rn = self.get_ngrams(reference_tokens, n)
            ngram_lists[n] = (Cn, Rn)
            all_ngrams_set.update(Cn)
            all_ngrams_set.update(Rn)

        if not all_ngrams_set:
            return 0.0

        all_ngrams = list(all_ngrams_set)
        emb_mat = self.get_bert_embeddings_batch(all_ngrams)  # [K, d], already L2-normalized
        idx_map = {ng: i for i, ng in enumerate(all_ngrams)}

        Pn_list: List[float] = []

        for n in range(1, N + 1):
            Cn, Rn = ngram_lists[n]

            if not Cn or not Rn:
                Pn_list.append(0.0)
                continue

            cand_idx = torch.tensor([idx_map[x] for x in Cn], device=self.device, dtype=torch.long)
            ref_idx = torch.tensor([idx_map[x] for x in Rn], device=self.device, dtype=torch.long)

            cand_embs = emb_mat[cand_idx]  # [C, d]
            ref_embs = emb_mat[ref_idx]    # [R, d]

            # Cosine similarity via dot product (embeddings unit-normalized)
            sims = ref_embs @ cand_embs.t()  # [R, C]
            max_sims, _ = sims.max(dim=1)    # best match for each reference n-gram
            Pn = float(max_sims.mean().item())
            Pn_list.append(Pn)

        Pmax = max(Pn_list) if Pn_list else 0.0

        # Length penalty (same as original)
        if Lr == 0:
            LP = 1.0
        else:
            length_diff = abs(Lc - Lr) / max(Lr, 1)
            if mode == "caption":
                LP = float(np.exp(-alpha * length_diff))
            elif mode == "semantic":
                LP = float(np.exp(alpha * (1.0 - length_diff)))
            else:
                LP = 1.0

        score = LP * Pmax
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_captioning(self, candidate: str, reference: str) -> float:
        cfg = self.captioning_cfg
        return self.bert_bleu_score(
            candidate,
            reference,
            N=cfg["N"],
            alpha=cfg["alpha"],
            mode=cfg["mode"],
        )

    # =====================
    # Polygon / IoU utils
    # =====================

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
        except Exception:
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

    # =====================
    # Grounding metric
    # =====================

    def evaluate_grounding(
        self,
        pred_boxes: List[List[float]],
        gt_boxes: List[List[float]],
    ) -> float:
        """
        Evaluate grounding using IoU and a count penalty.
        """
        cfg = self.grounding_cfg
        alpha = cfg["alpha"]
        coordinate_system = cfg["coordinate_system"]
        validate_bounds = cfg["validate_bounds"]
        iou_thresh = cfg["iou_threshold"]

        # Coordinate conversion
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

        # Optional bound validation
        if validate_bounds:
            for i, box in enumerate(pred_boxes):
                if not self.validate_obb_bounds(box, normalized=False):
                    print(f"Warning: Predicted box {i} is out of bounds")
            for i, box in enumerate(gt_boxes):
                if not self.validate_obb_bounds(box, normalized=False):
                    print(f"Warning: Ground truth box {i} is out of bounds")

        N_pred = len(pred_boxes)
        N_ref = len(gt_boxes)

        if N_pred == 0 and N_ref == 0:
            return 1.0
        if N_ref == 0 and N_pred > 0:
            return 0.0
        if N_pred == 0 and N_ref > 0:
            return 0.0

        # Count penalty
        count_penalty = float(np.exp(-alpha * abs(N_pred - N_ref)))

        # Greedy one-to-one matching by IoU
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
                if max_iou >= iou_thresh:
                    ious.append(max_iou)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        grounding_score = count_penalty * mean_iou

        return float(np.clip(grounding_score, 0.0, 1.0))

    # =====================
    # Binary metric
    # =====================

    def evaluate_binary(self, prediction: str, ground_truth: str) -> float:
        pred_normalized = prediction.strip().lower()
        gt_normalized = ground_truth.strip().lower()
        return 1.0 if pred_normalized == gt_normalized else 0.0

    # =====================
    # Numeric metric
    # =====================

    def evaluate_numeric(
        self,
        prediction: float,
        ground_truth: float,
        unit: str = None,
        prediction_unit: str = None,
    ) -> float:
        alpha = self.numeric_cfg["alpha"]

        # Unit conversion
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

        # Error and score mapping
        if ground_truth == 0:
            err = abs(prediction - ground_truth)
        else:
            err = abs(prediction - ground_truth) / abs(ground_truth)

        score = float(np.exp(-alpha * err))
        return float(np.clip(score, 0.0, 1.0))

    # =====================
    # Semantic (text) metric
    # =====================

    def evaluate_semantic(self, prediction: str, ground_truth: str) -> float:
        return self.bert_bleu_score(
            prediction,
            ground_truth,
            N=4,
            alpha=0.5,
            mode="semantic",
        )

    # =====================
    # Final score aggregation
    # =====================

    def compute_final_score(self, scores: Dict[str, float]) -> float:
        final_score = 0.0
        for task, weight in self.weights.items():
            if task in scores:
                final_score += weight * scores[task]
        return float(np.clip(final_score, 0, 1))

    # =====================
    # JSON helpers
    # =====================

    def parse_response_json(
        self,
        response_json: dict
    ) -> Tuple[Dict[str, Union[str, float, List]], Dict[str, Any]]:

        metadata = response_json.get('input_image', {}).get('metadata', {})

        self.image_width = metadata.get('width')
        self.image_height = metadata.get('height')
        self.spatial_resolution_m = metadata.get('spatial_resolution_m')

        predictions: Dict[str, Union[str, float, List]] = {}
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

    def load_predictions_from_json(
        self,
        json_path: str
    ) -> Tuple[Dict[str, Union[str, float, List]], Dict[str, Any]]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return self.parse_response_json(data)

    # =====================
    # Master evaluation
    # =====================

    def evaluate_all(
        self,
        predictions: Dict[str, Union[str, float, List]],
        ground_truths: Dict[str, Union[str, float, List]],
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        metadata = metadata or {}

        if 'captioning' in predictions and 'captioning' in ground_truths:
            scores['captioning'] = self.evaluate_captioning(
                predictions['captioning'], ground_truths['captioning']
            )

        if 'grounding' in predictions and 'grounding' in ground_truths:
            scores['grounding'] = self.evaluate_grounding(
                predictions['grounding'], ground_truths['grounding']
            )

        if 'binary' in predictions and 'binary' in ground_truths:
            scores['binary'] = self.evaluate_binary(
                predictions['binary'], ground_truths['binary']
            )

        if 'numeric' in predictions and 'numeric' in ground_truths:
            scores['numeric'] = self.evaluate_numeric(
                predictions['numeric'], ground_truths['numeric'],
                unit=self.numeric_cfg["default_unit"],
                prediction_unit=self.numeric_cfg["default_unit"],
            )

        if 'semantic' in predictions and 'semantic' in ground_truths:
            scores['semantic'] = self.evaluate_semantic(
                predictions['semantic'], ground_truths['semantic']
            )

        scores['final_score'] = self.compute_final_score(scores)
        return scores


def print_scores(scores: Dict[str, float], weights: Dict[str, float]) -> None:
    print("\nResults:")
    for task, score in scores.items():
        if task != 'final_score':
            weight = weights.get(task, 0)
            print(f"  {task.capitalize():15s}: {score:.4f} (weight: {weight:.0%})")
    print(f"  {'FINAL SCORE':15s}: {scores['final_score']:.4f}")


if __name__ == "__main__":
    print("Initializing GeoNLI Evaluator ...")
    print("=" * 60)

    metric_spec = {}
    evaluator = GeoNLIEvaluator(metric_spec=metric_spec)

    json_path = os.path.join(
        'sample_dataset_inter_iit_v1_3',
        'sample2_response.json'
    )
    predictions, metadata = evaluator.load_predictions_from_json(json_path)

    print(f"Device: {evaluator.device}")
    print(f"Spatial Resolution: {evaluator.spatial_resolution_m} meters/pixel")
    print(f"Image Dimensions: {evaluator.image_width} x {evaluator.image_height} pixels")
    print("OBB Format: [center_x, center_y, width, height, angle]")
    print("=" * 60)

    # Example: using predictions as GT for sanity check
    ground_truths = predictions.copy()

    scores = evaluator.evaluate_all(predictions, ground_truths, metadata)
    print_scores(scores, evaluator.weights)

    evaluator.log_to_wandb(
        scores, metadata, predictions, ground_truths,
        project_name="GeoNLI-Evaluation", run_name="sample2_run"
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
