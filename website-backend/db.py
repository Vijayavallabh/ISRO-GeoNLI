from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()


MONGODB_URL = os.getenv("DATABASE_URL")
print(MONGODB_URL)

client = AsyncIOMotorClient(MONGODB_URL)

db = client["isro-geo"] 

def get_collection(name: str):
    return db[name]
