from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SelectionRequest(BaseModel):
    selected_text: str
    question: str
    chapter: Optional[str] = None
    session_id: Optional[str] = None


class ChunkResult(BaseModel):
    content: str
    chapter: str
    section: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChunkResult]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    database: str
    timestamp: datetime
