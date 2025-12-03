from fastapi import APIRouter, Form, UploadFile, File, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from db import db
from dotenv import load_dotenv
from jose import jwt, JWTError
import os

router = APIRouter()

load_dotenv()

sessions_collection = db["sessions"]

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

@router.get("/chat-history")
async def chat_history(email: str = Depends(get_current_user_email) ):
    existing_sessions = await sessions_collection.find({"email":email}).to_list(length=None)
    session_list = [
        {"sessionId": session["sessionId"], "title": session["title"]}
        for session in existing_sessions
    ]
    return session_list
    