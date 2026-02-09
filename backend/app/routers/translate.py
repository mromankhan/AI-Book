from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.translate_service import translate_to_urdu, get_cached_translation

router = APIRouter(prefix="/api/translate")


class TranslateRequest(BaseModel):
    chapter: str
    content: str


@router.post("/urdu")
async def translate_urdu(req: TranslateRequest):
    result = await translate_to_urdu(req.chapter, req.content)
    return {"chapter": req.chapter, "translated_content": result, "language": "ur"}


@router.get("/urdu/{chapter}")
async def get_urdu_translation(chapter: str):
    cached = await get_cached_translation(chapter)
    if not cached:
        raise HTTPException(status_code=404, detail="No Urdu translation found for this chapter")
    return {"chapter": chapter, "translated_content": cached, "language": "ur"}
