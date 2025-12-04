import math
from typing import List, Tuple, Union, Dict, Any

def select_object_by_rank(objects_list: List[Dict], sort_key: str, rank_index: int, return_key: str = None):
    """
    Sorts objects by `sort_key`, but returns the value of `return_key`.
    
    Args:
        objects_list: List of object dicts from SAM.
        sort_key: Attribute to sort by (e.g., 'area_m2').
        rank_index: 1 (smallest), -1 (largest), etc.
        return_key: Attribute to return (e.g., 'shape'). If None, returns sort_key value.
    """
    # Filter objects that have the sort key
    valid_objects = [obj for obj in objects_list if sort_key in obj]
    
    if not valid_objects:
        return None, f"No objects found with attribute '{sort_key}'."

    # Sort ascending
    sorted_objs = sorted(valid_objects, key=lambda x: x[sort_key])

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
        
        # Determine what to return
        key_to_fetch = return_key if return_key else sort_key
        val = selected_obj.get(key_to_fetch, "N/A")
        obj_id = selected_obj.get('mask_id', 'unknown')
        
        # Format for float values to keep it clean
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)

        return selected_obj, f"{val_str}"
        
    except IndexError:
        return None, f"Rank {rank_index} is out of bounds. Only found {len(sorted_objs)} objects."

def filter_objects_by_region(objects_list: List[Dict], region: str, image_size: Tuple[int, int]):
    """
    Filters objects based on their centroid location within the image.
    Regions: top, bottom, left, right, top-left, top-right, bottom-left, bottom-right, center.
    """
    img_w, img_h = image_size
    filtered_objects = []
    
    for obj in objects_list:
        # Calculate centroid from coordinates [x1, y1, x2, y2...]
        coords = obj.get('coordinates', [])
        if not coords:
            continue
        cx = sum(coords[0::2]) / (len(coords) // 2)
        cy = sum(coords[1::2]) / (len(coords) // 2)
        
        match = False
        region = region.lower().replace("-", " ").strip()
        
        # Definitions
        is_top = cy < img_h / 2
        is_bottom = cy >= img_h / 2
        is_left = cx < img_w / 2
        is_right = cx >= img_w / 2
        
        if region == "top":
            match = is_top
        elif region == "bottom":
            match = is_bottom
        elif region == "left":
            match = is_left
        elif region == "right":
            match = is_right
        elif "top" in region and "left" in region:
            match = is_top and is_left
        elif "top" in region and "right" in region:
            match = is_top and is_right
        elif "bottom" in region and "left" in region:
            match = is_bottom and is_left
        elif "bottom" in region and "right" in region:
            match = is_bottom and is_right
        elif region == "center":
            # Central 50% of the image
            match = (img_w * 0.25 < cx < img_w * 0.75) and (img_h * 0.25 < cy < img_h * 0.75)
            
        if match:
            filtered_objects.append(obj)
            
    return filtered_objects

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


