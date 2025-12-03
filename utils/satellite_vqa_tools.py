import math
from typing import List, Tuple, Union, Dict, Any

def comparison_tool(data_pairs: List[Tuple[str, float]], descending: bool = False) -> List[Tuple[str, float]]:
    """
    Sorts a list of (id, value) pairs based on the numerical value.
    Useful for questions like "Which building is the largest?" or "List cars by size."

    Args:
        data_pairs: A list of tuples where index 0 is the Object ID and index 1 is the value (area, height, etc.).
        descending: If True, sorts from highest to lowest. Default is False (lowest to highest).

    Returns:
        A list of tuples sorted by the value.
    """
    # Sort based on the second element (the value)
    sorted_data = sorted(data_pairs, key=lambda x: x[1], reverse=descending)
    return sorted_data

def distance_tool(group_a: List[Tuple[str, float, float, float, float]], 
                  group_b: List[Tuple[str, float, float, float, float]]) -> List[Dict[str, Any]]:
    """
    Calculates pairwise Euclidean distances between objects in two lists.
    
    The input tuples represent bounding box coordinates: (id, x1, x2, y1, y2).
    This tool calculates the center point (centroid) of the box before measuring distance.

    Args:
        group_a: List of tuples (id, x1, x2, y1, y2).
        group_b: List of tuples (id, x1, x2, y1, y2).

    Returns:
        A list of dictionaries containing {'object_a': id, 'object_b': id, 'distance': float}.
    """
    results = []

    for item_a in group_a:
        id_a, cx_a, cy_a = item_a
        
        for item_b in group_b:
            id_b, cx_b, cy_b = item_b
            
            # If comparing a list to itself, skip comparing an object to itself
            if id_a == id_b:
                continue

            # Euclidean distance
            dist = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
            
            results.append({
                "from_obj": id_a,
                "to_obj": id_b,
                "distance": round(dist, 2)
            })

    # Sort results by distance (closest first) for convenience
    return sorted(results, key=lambda x: x['distance'])

def calculator_tool(expression: str) -> Union[float, str]:
    """
    Evaluates a mathematical expression to compute totals, ratios, etc.
    
    Args:
        expression: A string containing the math expression (e.g., "120.5 + 40", "sqrt(500)").
                    Supported operators: +, -, *, /, **, sqrt.

    Returns:
        The result of the calculation as a float, or an error message string.
    """
    # specific allowed names to prevent unsafe eval code injection
    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow
    }
    
    # Remove any dangerous characters or builtins not allowed
    # This is a basic sanitizer; for production, use a dedicated parser logic.
    safe_expression = expression.replace('^', '**')
    
    try:
        # Compile the code object restricted to eval mode
        code = compile(safe_expression, "<string>", "eval")
        
        # Verify no names are used that aren't in our allowed list
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")

        result = eval(code, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        return f"Error computing expression: {e}"

