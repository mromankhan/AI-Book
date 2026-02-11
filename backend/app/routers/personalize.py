from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.routers.auth import require_auth
from app.services.personalize_service import personalize_content, get_cached_personalization

router = APIRouter(prefix="/api/personalize")


class PersonalizeRequest(BaseModel):
    chapter: str
    content: str


@router.post("")
async def personalize(req: PersonalizeRequest, user: dict = Depends(require_auth)):
    result = await personalize_content(user, req.chapter, req.content)
    return {"chapter": req.chapter, "personalized_content": result}


@router.get("/{chapter}")
async def get_personalized(chapter: str, user: dict = Depends(require_auth)):
    cached = await get_cached_personalization(user["id"], chapter)
    if not cached:
        raise HTTPException(status_code=404, detail="No personalized content found for this chapter")
    return {"chapter": chapter, "personalized_content": cached}
