from openai import OpenAI
from typing import Optional

from app.config import settings
from app.models.database import get_pool

client = OpenAI(api_key=settings.openai_api_key)

PERSONALIZE_PROMPT = """You are a content personalization engine for an educational book on Physical AI & Humanoid Robotics.

Given the reader's background profile and a chapter's content, adapt the content to match their level:

Reader Profile:
- Programming Level: {programming_level}
- Hardware Experience: {hardware_experience}
- Education Level: {education_level}

Adaptation Rules:
- For "beginner" programming level: Use simpler language, add more analogies, explain jargon
- For "intermediate": Keep current depth, add practical tips
- For "advanced": Add deeper technical details, references to papers/specs
- For "none" hardware experience: Focus on software/simulation aspects
- For "some"/"extensive" hardware: Include hardware-specific guidance and real-world tips

IMPORTANT:
- Preserve the markdown formatting
- Keep all headings, code blocks, and structure intact
- Only adapt the explanatory text, not code examples
- Keep the same section structure
- Do NOT add or remove sections

Chapter content to personalize:
{content}
"""


async def get_cached_personalization(user_id: str, chapter: str) -> Optional[str]:
    p = await get_pool()
    async with p.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content FROM personalized_content WHERE user_id = $1 AND chapter = $2",
            user_id, chapter,
        )
    return row["content"] if row else None


async def save_personalization(user_id: str, chapter: str, content: str):
    p = await get_pool()
    async with p.acquire() as conn:
        await conn.execute(
            """INSERT INTO personalized_content (user_id, chapter, content)
               VALUES ($1, $2, $3)
               ON CONFLICT (user_id, chapter) DO UPDATE SET content = $3, created_at = NOW()""",
            user_id, chapter, content,
        )


async def personalize_content(user_profile: dict, chapter: str, content: str) -> str:
    # Check cache first
    user_id = user_profile.get("id", "")
    cached = await get_cached_personalization(user_id, chapter)
    if cached:
        return cached

    profile = user_profile.get("profile", {})
    prompt = PERSONALIZE_PROMPT.format(
        programming_level=profile.get("programming_level", "beginner"),
        hardware_experience=profile.get("hardware_experience", "none"),
        education_level=profile.get("education_level", "undergraduate"),
        content=content[:8000],  # Limit content length for API
    )

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": "You are a content personalization assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=4000,
    )

    personalized = response.choices[0].message.content

    # Cache result
    await save_personalization(user_id, chapter, personalized)

    return personalized
