from fastapi import APIRouter, Form, UploadFile, File, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from io import BytesIO
from datetime import datetime
from time import time
from db import db
import boto3, os
from dotenv import load_dotenv
from jose import jwt, JWTError
import requests
import base64
import json
import google.generativeai as genai
import httpx



# Usage example:
# api_key = "your-google-ai-studio-api-key"
# result = call_gemini_ai(api_key)
# print(result)


load_dotenv()

router = APIRouter()

sessions_collection = db["sessions"]
messages_collection = db["messages"]

api_key = os.getenv("GEMINI_API_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET = os.getenv("AWS_BUCKET_NAME")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()


async def get_current_user_email(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Email not found in token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def call_external_api(instruction: dict):
    url = "https://a2f7b83599770.notebooks.jarvislabs.net/proxy/8080/query"
    payload = instruction

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return {"result": response.json()}  # call .json()
    except httpx.HTTPStatusError as e:
        print("ERROR FROM EXTERNAL API:", e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=str("error"))
    except httpx.RequestError as e:
        print("REQUEST ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str("error ero"))

@router.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    query: str = Form(...),
    spatial_resolution_m: float = Form(...),
    image_url: str | None = Form(None),
    image: UploadFile | None = File(None),
    email: str = Depends(get_current_user_email) 
):
    if(image):
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        width, height = img.size

    existing_session = await sessions_collection.find_one({"sessionId": session_id})

    if not existing_session:
        file_name = f"{int(time())}_{image.filename}"
        s3_client.upload_fileobj(
            BytesIO(contents),
            BUCKET,
            file_name,
            ExtraArgs={"ACL": "public-read", "ContentType": image.content_type}
        )
        image_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_name}"
        await sessions_collection.insert_one({
            "sessionId": session_id,
            "email":email,
            "title": query,
            "imageURL": image_url,  
            "createdAt": int(time()),
        })

    await messages_collection.insert_one({
        "sessionId": session_id,
        "role": "user",
        "type":"text",
        "content": query,
        "timestamp": int(time())
    })

    result = await sessions_collection.find_one({"sessionId":session_id})
    image = result["imageURL"]
    print(image)

    response = requests.get(image)
    img = Image.open(BytesIO(response.content))
    width, height = img.size

    instruction = {
        "query":query,
        "image_url":image,
        "spatial_resolution_m":spatial_resolution_m
    }
    print(instruction)
    data = await call_external_api(instruction)

    print(data)

    response = data.get("result", {}).get("response", {})

    text = str(response.get("text", ""))       
    image_base64 = response.get("image", "")   


    ai_image_url=""
    if image_base64:
        base64_str = image_base64
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_str)
        file_name = f"ai_{int(time())}.png"
        s3_client.upload_fileobj(
            BytesIO(image_bytes),
            BUCKET,
            file_name,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/png"}
        )
        ai_image_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_name}"

    await messages_collection.insert_one({
        "sessionId": session_id,
        "role": "assistant",
        "type":"text",
        "content": text,
        "timestamp": int(time())
    })

    if ai_image_url:
        await messages_collection.insert_one({
            "sessionId": session_id,
            "role": "assistant",
            "type":"image",
            "content": ai_image_url,
            "timestamp": int(time())
        })
    print(ai_image_url, type(ai_image_url))
    print(type(text))
    #return {"reply": reply, "response": instruction}
    return {"text":text,"image":ai_image_url}
