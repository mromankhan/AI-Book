from openai import OpenAI
from typing import List

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    response = client.embeddings.create(
        input=text,
        model=settings.embedding_model
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    cleaned = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    if not cleaned:
        return []
    response = client.embeddings.create(
        input=cleaned,
        model=settings.embedding_model
    )
    return [item.embedding for item in response.data]
