from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr
from typing import Optional, List

from app.services.auth_service import (
    create_user, authenticate_user, get_user_with_profile,
    create_token, decode_token,
)

router = APIRouter(prefix="/api/auth")


class SignUpRequest(BaseModel):
    email: str
    password: str
    name: str
    programming_level: str = "beginner"
    hardware_experience: str = "none"
    education_level: str = "undergraduate"
    interests: List[str] = []


class SignInRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


async def get_current_user(request: Request) -> Optional[dict]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    user_id = decode_token(token)
    if not user_id:
        return None
    return await get_user_with_profile(user_id)


async def require_auth(request: Request) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignUpRequest):
    profile = {
        "programming_level": req.programming_level,
        "hardware_experience": req.hardware_experience,
        "education_level": req.education_level,
        "interests": req.interests,
    }

    user = await create_user(req.email, req.password, req.name, profile)
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")

    token = create_token(user["id"])
    return AuthResponse(token=token, user=user)


@router.post("/signin", response_model=AuthResponse)
async def signin(req: SignInRequest):
    user = await authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"])
    return AuthResponse(token=token, user=user)


@router.get("/me")
async def get_me(user: dict = Depends(require_auth)):
    return user
