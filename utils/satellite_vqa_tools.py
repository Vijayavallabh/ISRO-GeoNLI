import math
from typing import List, Tuple, Union, Dict, Any

def select_object_by_rank(objects_list: List[Dict], attribute_key: str, rank_index: int):
    """
    Sorts the object list by the given attribute and returns the object at rank_index.
    
    Args:
        objects_list: The list of object dicts from SAM state.
        attribute_key: The key to sort by (e.g., 'area_m2', 'confidence').
        rank_index: 1-based index. Positive for ascending (smallest first), 
                    Negative for descending (largest first).
                    e.g., 1 = smallest, -1 = largest, -2 = second largest.
    """
    # Filter out objects that might lack the key
    valid_objects = [obj for obj in objects_list if attribute_key in obj]
    
    if not valid_objects:
        return None, f"No objects found with attribute '{attribute_key}'."

    # Sort ascending
    sorted_objs = sorted(valid_objects, key=lambda x: x[attribute_key])

    
    try:
        if rank_index > 0:
            # 1st smallest -> index 0
            idx = rank_index - 1
            desc = "smallest"
        elif rank_index < 0:
            # -1 (largest) -> index -1
            idx = rank_index
            desc = "largest"
        else:
            return None, "Rank index cannot be 0."

        selected_obj = sorted_objs[idx]
        
        # We return the object, plus a pre-formatted string for the Agent to read
        val = selected_obj[attribute_key]
        obj_id = selected_obj.get('mask_id', 'unknown')
        
        return selected_obj, f"Found the {abs(rank_index)}-th {desc} object (ID: {obj_id}) with {attribute_key} = {val:.2f}."
        
    except IndexError:
        return None, f"Rank {rank_index} is out of bounds. Only found {len(sorted_objs)} objects."


def calculate_distance_by_indices(objects_list: List[Dict], idx1: int, idx2: int):
    """
    Calculates distance between two objects using their list indices.
    """
    if idx1 < 0 or idx1 >= len(objects_list) or idx2 < 0 or idx2 >= len(objects_list):
        return None

    obj_a = objects_list[idx1]
    obj_b = objects_list[idx2]

    
    def get_centroid(coords):
        xs = coords[0::2]
        ys = coords[1::2]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    cx_a, cy_a = get_centroid(obj_a['coordinates'])
    cx_b, cy_b = get_centroid(obj_b['coordinates'])
    
    dist_pixels = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
    return dist_pixels
    
def calculator_tool(expression: str) -> Union[float, str]:
    """
    Evaluates a mathematical expression (Same as before).
    """
    allowed_names = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "pow": pow}
    safe_expression = expression.replace('^', '**')
    try:
        code = compile(safe_expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        return f"Error computing expression: {e}"
