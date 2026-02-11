import asyncpg
import uuid
from datetime import datetime
from typing import Optional

from app.config import settings

pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)
        await _init_tables()
    return pool


async def _init_tables():
    p = await get_pool()
    async with p.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                page_context TEXT
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT REFERENCES chat_sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS book_chunks (
                id TEXT PRIMARY KEY,
                chapter TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                qdrant_point_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Auth tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY REFERENCES users(id),
                programming_level TEXT DEFAULT 'beginner',
                hardware_experience TEXT DEFAULT 'none',
                education_level TEXT DEFAULT 'undergraduate',
                preferred_language TEXT DEFAULT 'en',
                interests TEXT[],
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Cache tables for personalization and translation
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS personalized_content (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(id),
                chapter TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, chapter)
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS translated_content (
                id SERIAL PRIMARY KEY,
                chapter TEXT NOT NULL UNIQUE,
                language TEXT NOT NULL DEFAULT 'ur',
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)


async def create_session(page_context: Optional[str] = None) -> str:
    p = await get_pool()
    session_id = str(uuid.uuid4())
    async with p.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_sessions (id, page_context) VALUES ($1, $2)",
            session_id, page_context
        )
    return session_id


async def save_message(session_id: str, role: str, content: str, sources: Optional[str] = None):
    p = await get_pool()
    async with p.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources) VALUES ($1, $2, $3, $4::jsonb)",
            session_id, role, content, sources
        )


async def get_session_messages(session_id: str, limit: int = 10):
    p = await get_pool()
    async with p.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM chat_messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def close_pool():
    global pool
    if pool:
        await pool.close()
        pool = None
