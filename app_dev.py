"""
DEVELOPMENT API Server
Fast startup with lazy model loading and auto-reload support.

Run: uvicorn app_dev:app --reload --port 8000
"""

from typing import Optional, Dict, Any
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from utils.visualization import annotate_image_with_boxes
from api_helpers import (
    get_pipeline,
    classify_query,
    strip_data_prefix,
    decode_image_from_base64,
    get_image_from_input,
    image_to_base64,
    format_grounding_response,
    normalize_vqa_answer
)
from api_models import (
    ImageMetadata,
    InputImage,
    CaptionQuery,
    GroundingQuery,
    AttributeQuery,
    Queries,
    StructuredRequest,
    SimpleRequest
)

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG while developing
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename='app.log'
)


# ============================================================================
# FastAPI App - Development Configuration
# ============================================================================

app = FastAPI(
    title="ISRO-GeoNLI API [DEV]",
    description="Remote Sensing image analysis (captioning, grounding, VQA) - Development Mode",
    version="1.0.0-dev",
    debug=True
)


# Startup event: NO preload in dev (faster startup)
@app.on_event("startup")
async def startup_event():
    """Development mode: skip model preloading for faster startup."""
    print("\n" + "="*60)
    print("🚀 STARTING SERVER - DEVELOPMENT MODE")
    print("="*60)
    print("   Environment: development")
    print("   Auto-reload: enabled")
    print("   Model loading: lazy (on first request)")
    print("   Running on: http://127.0.0.1:8000")
    print("="*60 + "\n")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "ISRO-GeoNLI API [Development]",
        "environment": "development",
        "version": "2.0.0-dev",
        "endpoints": {
            "unified": "/process",
            "simple": "/query",
            "individual": ["/caption", "/grounding", "/vqa"],
            "legacy": ["/process-form", "/process-json"],
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    from api_helpers import _pipeline
    is_ready = _pipeline is not None
    
    status = {
        "status": "healthy" if is_ready else "starting",
        "environment": "development",
        "pipeline_loaded": is_ready
    }
    
    if is_ready:
        status["device"] = str(_pipeline.device)
        status["models"] = {
            "vlm": _pipeline.vlm is not None,
            "sam3": _pipeline.sam3 is not None
        }
    
    return status


# ============================================================================
# NEW UNIFIED ENDPOINT - Accepts structured schema
# ============================================================================

@app.post("/process")
async def process_structured(request: StructuredRequest, include_annotations: bool = False):
    """
    Process structured request matching query.json schema.
    Executes each query type individually through the pipeline.
    """
    pipeline = pipeline = get_pipeline(vlm_model_id="Dinosaur2314/qwen_finetune11")

    # Load image
    image = get_image_from_input(request.input_image)

    # Target response schema
    results = {
        "input_image": {
            "image_id": request.input_image.image_id,
            "image_url": request.input_image.image_url,
            "metadata": (
                request.input_image.metadata.dict()
                if request.input_image.metadata
                else None
            )
        },
        "queries": {}
    }

    # Caption Query
    if request.queries.caption_query:
        instruction = request.queries.caption_query.instruction
        caption = pipeline.generate_caption(image, instruction)

        results["queries"]["caption_query"] = {
            "instruction": instruction,
            "response": caption
        }

    # Grounding Query
    if request.queries.grounding_query:
        instruction = request.queries.grounding_query.instruction
        detections = pipeline.ground_objects(image, instruction)

        annotated_image = None
        if include_annotations and detections:
            obbs = [det["obbox"] for det in detections]
            annotated_img, _ = annotate_image_with_boxes(image.copy(), obbs)
            annotated_image = image_to_base64(annotated_img)

        results["queries"]["grounding_query"] = {
            "instruction": instruction,
            "response": format_grounding_response(detections),
            **({"annotated_image": annotated_image} if include_annotations else {})
        }

    # Attribute Queries
    if request.queries.attribute_query:
        attr_results = {}

        if request.queries.attribute_query.binary:
            instruction = request.queries.attribute_query.binary["instruction"]
            raw_answer = pipeline.answer_question(
                image, instruction, question_type="binary"
            )
            answer = normalize_vqa_answer(raw_answer, "binary")

            attr_results["binary"] = {
                "instruction": instruction,
                "response": answer
            }

        if request.queries.attribute_query.numeric:
            instruction = request.queries.attribute_query.numeric["instruction"]
            raw_answer = pipeline.answer_question(
                image, instruction, question_type="numeric"
            )
            answer = normalize_vqa_answer(raw_answer, "numeric")

            attr_results["numeric"] = {
                "instruction": instruction,
                "response": answer
            }

        if request.queries.attribute_query.semantic:
            instruction = request.queries.attribute_query.semantic["instruction"]
            answer = pipeline.answer_question(
                image, instruction, question_type="semantic"
            )

            attr_results["semantic"] = {
                "instruction": instruction,
                "response": answer
            }

        results["queries"]["attribute_query"] = attr_results

    return results


# ============================================================================
# SIMPLE QUERY ENDPOINT - Auto-classifies query type
# ============================================================================

@app.post("/query")
async def process_simple_query(request: SimpleRequest):
    """
    Simple query endpoint with LLM-based classification.
    Accepts plain text query and classifies it into structured format.
    """
    # Build input image first
    input_image = InputImage(
        image_id="query_image",
        image_url=request.image_url,
        image_base64=request.image_base64,
        image_path=request.image_path
    )
    
    # Load the actual image
    image = get_image_from_input(input_image)
    
    # Classify the query with the actual image for better context
    classified_queries = classify_query(request.query, image=image)
    
    queries = Queries(
        caption_query=CaptionQuery(**classified_queries["caption_query"]) if classified_queries["caption_query"] else None,
        grounding_query=GroundingQuery(**classified_queries["grounding_query"]) if classified_queries["grounding_query"] else None,
        attribute_query=AttributeQuery(**classified_queries["attribute_query"]) if classified_queries["attribute_query"] else None
    )
    
    structured = StructuredRequest(input_image=input_image, queries=queries)
    
    # Process through unified endpoint
    structured_response = await process_structured( structured, include_annotations=True )

    response_text = ""
    response_image = ""
    queries_out = structured_response.get("queries", {})

    if "attribute_query" in queries_out:
        attr = queries_out["attribute_query"]
        for _, v in attr.items():
            response_text = str(v.get("response", ""))
            break

    elif "caption_query" in queries_out:
        response_text = queries_out["caption_query"].get("response", "")

    elif "grounding_query" in queries_out:
        grounding = queries_out["grounding_query"]
        response_text = grounding.get("response", [])
        response_image = grounding.get("annotated_image", "") or ""

    return {
        "query": request.query,
        "response": {
            "text": response_text,
            "image": response_image
        }
    }


# ============================================================================
# LEGACY ENDPOINTS - Backward compatibility
# ============================================================================

@app.post("/process-form")
async def process_form(
    text: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    """Accept form with text and image, return normalized JSON."""
    if image_file:
        content = await image_file.read()
        b64 = base64.b64encode(content).decode("utf-8")
    else:
        b64 = strip_data_prefix(image_base64 or "")

    if b64:
        try:
            base64.b64decode(b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    return {"text": text, "image": b64}


@app.post("/process-json")
async def process_json(payload: dict):
    """Accept JSON with text and image, return normalized JSON."""
    text = payload.get("text", "")
    image_raw = payload.get("image", "")
    b64 = strip_data_prefix(image_raw)

    if b64:
        try:
            base64.b64decode(b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    return {"text": text, "image": b64}


@app.post("/caption")
async def caption_endpoint(payload: dict):
    """Generate image caption using the pipeline."""
    instruction = payload.get("text", "")
    image_raw = payload.get("image", "")
    vlm_model_id = payload.get("vlm_model_id")

    b64 = strip_data_prefix(image_raw)
    if not b64:
        raise HTTPException(status_code=400, detail="Missing image data")

    image = decode_image_from_base64(b64)

    try:
        pipeline = get_pipeline(vlm_model_id=vlm_model_id) if vlm_model_id else get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        caption = pipeline.generate_caption(image, instruction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {e}")

    return {"text": instruction, "image": b64, "caption": caption}


@app.post("/grounding")
async def grounding_endpoint(payload: dict):
    """Run object detection/grounding using the pipeline."""
    query = payload.get("text", "")
    image_raw = payload.get("image", "")
    gsd = float(payload.get("gsd", 1.0))
    score_threshold = float(payload.get("score_threshold", 0.4))

    b64 = strip_data_prefix(image_raw)
    if not b64:
        raise HTTPException(status_code=400, detail="Missing image data")

    image = decode_image_from_base64(b64)

    try:
        pipeline = get_pipeline(vlm_model_id="Dinosaur2314/qwen_finetune11")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        detections = pipeline.ground_objects(image, query, gsd=gsd, score_threshold=score_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding failed: {e}")

    grounding_response = format_grounding_response(detections)
    return {"text": query, "image": b64, "detections": grounding_response}


@app.post("/vqa")
async def vqa_endpoint(payload: dict):
    """Run Visual Question Answering using the pipeline."""
    question = payload.get("text", "")
    image_raw = payload.get("image", "")
    question_type = payload.get("question_type", "numeric")
    gsd = float(payload.get("gsd", 1.0))
    detections = payload.get("detections")

    b64 = strip_data_prefix(image_raw)
    if not b64:
        raise HTTPException(status_code=400, detail="Missing image data")

    image = decode_image_from_base64(b64)

    try:
        pipeline = get_pipeline(vlm_model_id="Dinosaur2314/qwen_finetune11")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert grounding response format to pipeline format if needed
    if detections and isinstance(detections, list):
        parsed_dets = []
        for d in detections:

            if "obbox" in d:
                obbox = d["obbox"]

                # Case 1: already 8-point polygon 
                if isinstance(obbox, (list, tuple)) and len(obbox) == 8:
                    polygon = list(map(float, obbox))

                # Case 2: angle format → convert BACK to 8 points
                elif isinstance(obbox, (list, tuple)) and len(obbox) == 3:
                    # RotatedRect -> polygon
                    box_points = cv2.boxPoints(tuple(obbox))
                    polygon = box_points.reshape(-1).tolist()

                else:
                    raise ValueError(f"Invalid obbox format: {obbox}")

                parsed_dets.append({
                    "id": d.get("object-id", ""),
                    "obbox": polygon
                })

            else:
                raise KeyError(f"No obbox found in detection: {d}")

        detections = parsed_dets

    try:
        answer = pipeline.answer_question(image=image, query=question, detections=detections, question_type=question_type, gsd=gsd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VQA failed: {e}")

    return {"text": question, "image": b64, "answer": answer}
