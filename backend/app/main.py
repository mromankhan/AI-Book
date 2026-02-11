from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.database import get_pool, close_pool
from app.routers import chat, health, auth, personalize, translate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database pool
    await get_pool()
    yield
    # Shutdown: close database pool
    await close_pool()


app = FastAPI(
    title="Physical AI Book - RAG Chatbot API",
    description="RAG-powered chatbot for the Physical AI: Humanoid & Robotics Systems book",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(personalize.router)
app.include_router(translate.router)


@app.get("/")
async def root():
    return {
        "name": "Physical AI Book RAG Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
    }
