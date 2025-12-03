from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from db import get_collection
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
users_col = get_collection("users")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/login")
async def login(req: LoginRequest):

    user = await users_col.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if req.password != user["password"]:
        raise HTTPException(status_code=400, detail="Incorrect password")

    payload = {"email": user["email"]}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {"access_token": token, "token_type": "bearer"}
