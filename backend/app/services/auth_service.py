import uuid
from datetime import datetime, timedelta
from typing import Optional

from passlib.context import CryptContext
from jose import JWTError, jwt

from app.config import settings
from app.models.database import get_pool

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiry_hours)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None


async def create_user(email: str, password: str, name: str, profile: dict) -> dict:
    p = await get_pool()
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)

    async with p.acquire() as conn:
        # Check if email exists
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
        if existing:
            return None

        await conn.execute(
            "INSERT INTO users (id, email, password_hash, name) VALUES ($1, $2, $3, $4)",
            user_id, email, password_hash, name,
        )
        await conn.execute(
            """INSERT INTO user_profiles (user_id, programming_level, hardware_experience, education_level, interests)
               VALUES ($1, $2, $3, $4, $5)""",
            user_id,
            profile.get("programming_level", "beginner"),
            profile.get("hardware_experience", "none"),
            profile.get("education_level", "undergraduate"),
            profile.get("interests", []),
        )

    return {"id": user_id, "email": email, "name": name}


async def authenticate_user(email: str, password: str) -> Optional[dict]:
    p = await get_pool()
    async with p.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, name, password_hash FROM users WHERE email = $1", email
        )

    if not row or not verify_password(password, row["password_hash"]):
        return None

    return {"id": row["id"], "email": row["email"], "name": row["name"]}


async def get_user_with_profile(user_id: str) -> Optional[dict]:
    p = await get_pool()
    async with p.acquire() as conn:
        user = await conn.fetchrow("SELECT id, email, name FROM users WHERE id = $1", user_id)
        if not user:
            return None
        profile = await conn.fetchrow(
            "SELECT programming_level, hardware_experience, education_level, interests FROM user_profiles WHERE user_id = $1",
            user_id,
        )

    result = {"id": user["id"], "email": user["email"], "name": user["name"]}
    if profile:
        result["profile"] = {
            "programming_level": profile["programming_level"],
            "hardware_experience": profile["hardware_experience"],
            "education_level": profile["education_level"],
            "interests": list(profile["interests"]) if profile["interests"] else [],
        }
    return result
