from fastapi import APIRouter, Form, UploadFile, File, Query
from db import db
import os
from fastapi.encoders import jsonable_encoder

router = APIRouter()

messages = db["messages"]
sessions = db["sessions"]

@router.get("/get-history")
async def getHistory(sessionId : str = Query(...)):
    message = await messages.find({"sessionId":sessionId}).to_list(length=None)
    message_list = [
        {"role": msg["role"],"type":msg["type"],"content": msg["content"]}
        for msg in message
    ]
    image = await sessions.find_one({"sessionId":sessionId})
    image_url = {"image_url": image["imageURL"]}

    return jsonable_encoder({"messages":message_list,"image":image_url})
    #return message_list