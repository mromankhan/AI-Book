from openai import OpenAI
from typing import Optional

from app.config import settings
from app.models.database import get_pool

client = OpenAI(api_key=settings.openai_api_key)

TRANSLATE_PROMPT = """Translate the following educational content about Physical AI and Robotics from English to Urdu.

Rules:
- Translate all text to Urdu
- Keep technical terms in English with Urdu transliteration in parentheses where helpful
- Preserve all markdown formatting (headings, code blocks, lists, bold, italic)
- Do NOT translate code blocks — keep them in English
- Keep proper nouns (ROS 2, NVIDIA, Gazebo, etc.) in English
- Maintain the same document structure
- Use formal Urdu appropriate for educational content

Content to translate:
{content}
"""


async def get_cached_translation(chapter: str, language: str = "ur") -> Optional[str]:
    p = await get_pool()
    async with p.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content FROM translated_content WHERE chapter = $1 AND language = $2",
            chapter, language,
        )
    return row["content"] if row else None


async def save_translation(chapter: str, content: str, language: str = "ur"):
    p = await get_pool()
    async with p.acquire() as conn:
        await conn.execute(
            """INSERT INTO translated_content (chapter, language, content)
               VALUES ($1, $2, $3)
               ON CONFLICT (chapter) DO UPDATE SET content = $3, created_at = NOW()""",
            chapter, language, content,
        )


async def translate_to_urdu(chapter: str, content: str) -> str:
    # Check cache first
    cached = await get_cached_translation(chapter)
    if cached:
        return cached

    prompt = TRANSLATE_PROMPT.format(content=content[:8000])

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": "You are an expert English to Urdu translator specializing in technical and educational content."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
    )

    translated = response.choices[0].message.content

    # Cache result
    await save_translation(chapter, translated)

    return translated
