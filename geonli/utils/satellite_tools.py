"""
Tool implementations for the Satellite VQA Agent.
These are backend-agnostic and do NOT depend on the old codebase.
"""

import math
from typing import Dict, List, Union

ATTRIBUTE_MAP = {
    "area": "area_m2",
    "width": "width_m",
    "height": "height_m",
    "orientation": "orientation_deg",
    "shape": "shape",
    "confidence": "confidence",
    "length": "height_m",
    "perimeter": None,
}


def get_attribute_value(obj: Dict, attribute: str) -> Union[float, str, None]:
    if attribute not in ATTRIBUTE_MAP:
        return None
    mapped_key = ATTRIBUTE_MAP[attribute]
    if mapped_key and mapped_key in obj:
        return obj[mapped_key]
    if attribute == "perimeter":
        w = obj.get("width_m", 0)
        h = obj.get("height_m", 0)
        return 2 * (w + h) if w and h else None
    return None


def select_object_by_rank(
    objects_list: List[Dict],
    sort_key: str,
    rank_index: int,
    return_key: str = None,
) -> tuple:
    sort_attribute = ATTRIBUTE_MAP.get(sort_key)
    if not sort_attribute:
        return None, f"Invalid sort attribute: '{sort_key}'"

    valid_objects = [obj for obj in objects_list if sort_attribute in obj]
    if not valid_objects:
        return None, f"No objects found with attribute '{sort_key}'"

    sorted_objs = sorted(valid_objects, key=lambda x: x[sort_attribute])

    try:
        if rank_index > 0:
            idx = rank_index - 1
        elif rank_index < 0:
            idx = rank_index
        else:
            return None, "Rank index cannot be 0"

        selected_obj = sorted_objs[idx]
        return_attribute = return_key if return_key else sort_key
        value = get_attribute_value(selected_obj, return_attribute)

        if value is None:
            return None, f"Attribute '{return_attribute}' not available"

        if isinstance(value, float):
            return selected_obj, f"{value:.2f}"
        return selected_obj, str(value)
    except IndexError:
        return None, f"Rank {rank_index} out of bounds (only {len(sorted_objs)} objects)"


def calculate_distance_by_indices(
    objects_list: List[Dict],
    idx1: int,
    idx2: int,
    gsd: float = 1.0,
) -> Union[float, str]:
    if idx1 < 0 or idx1 >= len(objects_list):
        return f"Index {idx1} out of bounds"
    if idx2 < 0 or idx2 >= len(objects_list):
        return f"Index {idx2} out of bounds"

    coords_a = objects_list[idx1].get("coordinates", [])
    coords_b = objects_list[idx2].get("coordinates", [])

    if len(coords_a) < 8 or len(coords_b) < 8:
        return "Missing coordinate data"

    def centroid(coords):
        xs = coords[0::2]
        ys = coords[1::2]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    cx_a, cy_a = centroid(coords_a)
    cx_b, cy_b = centroid(coords_b)
    dist_pixels = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    return dist_pixels * gsd


def calculator_tool(expression: str) -> Union[float, str]:
    allowed = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
        "len": len,
    }
    safe_expr = expression.replace("^", "**")
    try:
        code = compile(safe_expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed:
                raise NameError(f"Use of '{name}' not allowed")
        result = eval(code, {"__builtins__": {}}, allowed)
        return float(result)
    except Exception as e:
        return f"Error: {e}"


def get_available_attributes() -> List[str]:
    return list(ATTRIBUTE_MAP.keys())
