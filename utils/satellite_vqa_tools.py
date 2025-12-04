import math
from typing import List, Dict, Union

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
    """
    Safely retrieve attribute from object metadata.
    Handles computed attributes and missing data.
    """
    if attribute not in ATTRIBUTE_MAP:
        return None
    
    mapped_key = ATTRIBUTE_MAP[attribute]
    
    # Direct attribute lookup
    if mapped_key and mapped_key in obj:
        return obj[mapped_key]
    
    # Computed attributes
    if attribute == "perimeter":
        # Approximate perimeter from width/height (rectangle assumption)
        w = obj.get("width_m", 0)
        h = obj.get("height_m", 0)
        return 2 * (w + h) if w and h else None
    
    return None
    

def select_object_by_rank(
    objects_list: List[Dict], 
    sort_key: str, 
    rank_index: int, 
    return_key: str = None
) -> tuple[Dict, str]:
    """
    Sorts objects by an attribute and returns a specific ranked object's attribute.
    
    Args:
        objects_list: List of object metadata dicts from SAM
        sort_key: Attribute to sort by (user-facing name like 'area', 'width')
        rank_index: 1=smallest, -1=largest, 2=second smallest, etc.
        return_key: Attribute to return (if None, returns sort_key value)
    
    Returns:
        (selected_object, formatted_value_string)
    """
    sort_attribute = ATTRIBUTE_MAP.get(sort_key)
    if not sort_attribute:
        return None, f"Invalid sort attribute: '{sort_key}'"
    
    valid_objects = [obj for obj in objects_list if sort_attribute in obj]
    
    if not valid_objects:
        return None, f"No objects found with attribute '{sort_key}'"
    
    sorted_objs = sorted(valid_objects, key=lambda x: x[sort_attribute])
    
    try:
        # Handle positive and negative indices
        if rank_index > 0:
            idx = rank_index - 1  # 1-indexed to 0-indexed
        elif rank_index < 0:
            idx = rank_index  # -1 is already correct for last element
        else:
            return None, "Rank index cannot be 0"
        
        selected_obj = sorted_objs[idx]
        
        # Determine what attribute to return
        return_attribute = return_key if return_key else sort_key
        value = get_attribute_value(selected_obj, return_attribute)
        
        if value is None:
            return None, f"Attribute '{return_attribute}' not available"
        
        # Format output based on type
        if isinstance(value, float):
            return selected_obj, f"{value:.2f}"
        else:
            return selected_obj, str(value)
        
    except IndexError:
        return None, f"Rank {rank_index} out of bounds (only {len(sorted_objs)} objects)"


def filter_objects_by_region(
    objects_list: List[Dict], 
    region: str, 
    image_size: tuple
) -> List[Dict]:
    """
    Filter objects by spatial region in the image.
    
    Args:
        objects_list: List of object metadata
        region: 'top', 'bottom', 'left', 'right', 'center', 'top-left', etc.
        image_size: (width, height) of image
    
    Returns:
        Filtered list of objects
    """
    img_w, img_h = image_size
    filtered = []
    
    region = region.lower().replace("-", " ").replace("_", " ").strip()
    
    for obj in objects_list:
        coords = obj.get('coordinates', [])
        if len(coords) < 8:
            continue
        
        # Calculate centroid
        cx = sum(coords[0::2]) / 4
        cy = sum(coords[1::2]) / 4
        
        # Region checks
        is_top = cy < img_h / 2
        is_bottom = cy >= img_h / 2
        is_left = cx < img_w / 2
        is_right = cx >= img_w / 2
        
        match = False
        
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
            match = (img_w * 0.25 < cx < img_w * 0.75) and \
                    (img_h * 0.25 < cy < img_h * 0.75)
        
        if match:
            filtered.append(obj)
    
    return filtered


def calculate_distance_by_indices(
    objects_list: List[Dict], 
    idx1: int, 
    idx2: int,
    gsd: float = 1.0
) -> Union[float, str]:
    """
    Calculate Euclidean distance between centroids of two objects.
    
    Args:
        objects_list: List of object metadata from SAM
        idx1, idx2: 0-based indices of objects
        gsd: Ground sampling distance (meters/pixel)
    
    Returns:
        Distance in meters, or error string
    """
    if idx1 < 0 or idx1 >= len(objects_list):
        return f"Index {idx1} out of bounds"
    if idx2 < 0 or idx2 >= len(objects_list):
        return f"Index {idx2} out of bounds"
    
    obj_a = objects_list[idx1]
    obj_b = objects_list[idx2]
    
    # Extract coordinates (8-point OBB: [x1,y1,x2,y2,x3,y3,x4,y4])
    coords_a = obj_a.get('coordinates', [])
    coords_b = obj_b.get('coordinates', [])
    
    if len(coords_a) < 8 or len(coords_b) < 8:
        return "Missing coordinate data"
    
    # Calculate centroids
    def get_centroid(coords):
        xs = coords[0::2]  # Even indices
        ys = coords[1::2]  # Odd indices
        return sum(xs) / len(xs), sum(ys) / len(ys)
    
    cx_a, cy_a = get_centroid(coords_a)
    cx_b, cy_b = get_centroid(coords_b)
    
    # Euclidean distance in pixels
    dist_pixels = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
    
    # Convert to meters
    dist_meters = dist_pixels * gsd
    
    return dist_meters
    
    
def calculator_tool(expression: str) -> Union[float, str]:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Math string like "(100 + 200) / 2"
    
    Returns:
        Computed value or error message
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
        "len": len,
    }
    
    safe_expression = expression.replace('^', '**')
    
    try:
        # Compile to check for forbidden names
        code = compile(safe_expression, "<string>", "eval")
        
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' not allowed")
        
        # Evaluate safely
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return float(result)
        
    except Exception as e:
        return f"Error: {e}"

def get_available_attributes() -> List[str]:
    """Returns list of all queryable attributes."""
    return list(ATTRIBUTE_MAP.keys())






