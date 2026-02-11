import json
from typing import List, Optional, AsyncGenerator

from openai import OpenAI

from app.config import settings
from app.services.embeddings import get_embedding
from app.services.qdrant_service import search_similar
from app.models.database import get_session_messages

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are an expert AI tutor for the book "Physical AI: Humanoid & Robotics Systems".
Your role is to help readers understand the book's content about Physical AI, robotics, ROS 2,
simulation, NVIDIA Isaac, and humanoid robots.

Rules:
- Answer questions based ONLY on the provided book context below.
- If the answer is not found in the context, say "I couldn't find information about that in the book.
  Try asking about topics covered in the chapters."
- Always reference the relevant chapter and section in your answer.
- Use clear, beginner-friendly language.
- Format your responses with markdown for readability.
- Keep answers concise but thorough.
"""

SELECTION_SYSTEM_PROMPT = """You are an expert AI tutor for the book "Physical AI: Humanoid & Robotics Systems".
The reader has selected a specific passage from the book and has a question about it.

Rules:
- Answer based primarily on the selected text passage provided.
- You may use additional book context to provide deeper explanations.
- Reference the chapter/section of the selected text.
- Use clear, beginner-friendly language.
- Format responses with markdown.
"""


def _build_context(chunks: List[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Chapter: {chunk['chapter']} | Section: {chunk['section']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def _build_sources(chunks: List[dict]) -> List[dict]:
    seen = set()
    sources = []
    for chunk in chunks:
        key = f"{chunk['chapter']}:{chunk['section']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "content": chunk["content"][:200] + "...",
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "score": chunk["score"],
            })
    return sources


async def rag_chat(
    question: str,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    # Get conversation history for context
    history = []
    if session_id:
        history = await get_session_messages(session_id, limit=6)

    # Embed the question
    query_vector = get_embedding(question)

    # Search for relevant chunks
    chunks = search_similar(query_vector, limit=5)
    context = _build_context(chunks)
    sources = _build_sources(chunks)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add context
    messages.append({
        "role": "system",
        "content": f"Book context for answering the question:\n\n{context}"
    })

    # Add conversation history
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    # Stream response
    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=1000,
    )

    # First yield sources as a JSON event
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # Then stream the answer
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def rag_selection_chat(
    selected_text: str,
    question: str,
    chapter: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    # Embed the question for additional context retrieval
    combined_query = f"{question} {selected_text[:200]}"
    query_vector = get_embedding(combined_query)

    # Search for related chunks (augment with vector search)
    chunks = search_similar(query_vector, limit=3, chapter_filter=chapter)
    additional_context = _build_context(chunks) if chunks else ""
    sources = _build_sources(chunks)

    # Build messages
    messages = [{"role": "system", "content": SELECTION_SYSTEM_PROMPT}]

    messages.append({
        "role": "system",
        "content": f"Selected text from the book:\n\n{selected_text}"
    })

    if additional_context:
        messages.append({
            "role": "system",
            "content": f"Additional book context:\n\n{additional_context}"
        })

    messages.append({"role": "user", "content": question})

    # Stream response
    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=1000,
    )

    # First yield sources
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
