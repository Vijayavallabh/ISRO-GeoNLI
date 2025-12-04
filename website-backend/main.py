from fastapi import FastAPI
from routers.chat import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from routers.login import router as login_router
from routers.chatHistory import router as history_router
from routers.getHistory import router as messages_router

app = FastAPI()

origins = [
    "https://isrogeonli.in:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(login_router, prefix="/api", tags=["Auth"])
app.include_router(history_router, prefix="/api", tags=["History"])
app.include_router(messages_router, prefix="/api", tags=["messages"])

@app.get("/")

def root():
    return {"message":"FastAPI is running."}