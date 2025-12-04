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

load_dotenv()

router = APIRouter()

sessions_collection = db["sessions"]
messages_collection = db["messages"]

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

AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8080")

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

    existing_session = await sessions_collection.find_one({"sessionId": session_id})

    if not existing_session:
        if not image:
            raise HTTPException(status_code=400, detail="Image required for new session")
        
        image_filename = image.filename or 'image.jpg'
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
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    image_url_db = result["imageURL"]
    if not image_url_db:
        raise HTTPException(status_code=400, detail="No image URL found for session")

    ai_request = {
        "query": query,
        "image_url": image_url_db
    }
    bot_text = "Sorry, I couldn't process that."
    bot_image_url = None

    try:
        print(f"Calling AI server /query endpoint with query: {query[:50]}...")
        ai_response = requests.post(
            f"{AI_SERVER_URL}/query",
            json=ai_request,
            timeout=240  # officially, 3 minute window to get responses
        )
        ai_response.raise_for_status()
        ai_data = ai_response.json()

        results = ai_data.get("results", {})

        if "caption" in results:
            bot_text = results["caption"]["response"]
        
        elif "grounding" in results:
            grounding = results["grounding"]
            box_text = grounding.get("detections", [])
            
            # If AI returned an annotated image (base64), upload it to S3
            annotated_image_b64 = grounding.get("annotated_image")
            if annotated_image_b64:
                # Strip data URL prefix if present
                if "base64," in annotated_image_b64:
                    b64_str = annotated_image_b64.split("base64,")[1]
                else:
                    b64_str = annotated_image_b64

                try:
                    annotated_bytes = base64.b64decode(b64_str)
                    file_name = f"ai_grounding_{int(time())}.png"
                    
                    s3_client.upload_fileobj(
                        BytesIO(annotated_bytes),
                        BUCKET,
                        file_name,
                        ExtraArgs={"ACL": "public-read", "ContentType": "image/png"}
                    )
                    bot_image_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_name}"
                except Exception as e:
                    print(f"Error uploaing annotate image to S3: {e}")
            
        elif "attributes" in results:
            # Check which attribute type has a response
            attrs = results["attributes"]
            if "binary" in attrs:
                bot_text = str(attrs["binary"]["response"])
            elif "numeric" in attrs:
                bot_text = str(attrs["numeric"]["response"])
            elif "semantic" in attrs:
                bot_text = str(attrs["semantic"]["response"])

    
    except requests.exceptions.Timeout:
        print(f"AI Server timeout after 240 seconds")
        bot_text = "The AI model is taking too long to respond. Please try again."
    
        except requests.exceptions.ConnectionError:
        print(f"AI Server connection error")
        bot_text = "Unable to connect to AI server. Please check if the server is running."

    except requests.exceptions.RequestException as e:
        print(f"AI Server Error: {e}")
        bot_text = f"Error connecting to AI model: {str(e)}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        bot_text = f"Error processing request: {str(e)}"

    await messages_collection.insert_one({
        "sessionId": session_id,
        "role": "assistant",
        "type":"text",
        "content": bot_text,
        "timestamp": int(time())
    })
    if bot_image_url:
        await messages_collection.insert_one({
            "sessionId": session_id,
            "role": "assistant",
            "type":"image",
            "content": bot_image_url,
            "timestamp": int(time())
        })

    return {"reply": bot_text,"image":bot_image_url}
