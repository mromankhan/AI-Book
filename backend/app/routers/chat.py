import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, SelectionRequest
from app.models.database import create_session, save_message
from app.services.rag_service import rag_chat, rag_selection_chat

router = APIRouter()


@router.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        session_id = await create_session()

    # Save user message
    await save_message(session_id, "user", request.message)

    async def stream_wrapper():
        full_response = []
        sources = []

        async for event in rag_chat(request.message, session_id):
            yield event
            # Parse for saving
            if event.startswith("data: "):
                try:
                    data = json.loads(event[6:].strip())
                    if data["type"] == "token":
                        full_response.append(data["content"])
                    elif data["type"] == "sources":
                        sources = data["sources"]
                except (json.JSONDecodeError, KeyError):
                    pass

        # Save assistant response
        response_text = "".join(full_response)
        await save_message(session_id, "assistant", response_text, json.dumps(sources) if sources else None)

        # Send session_id as final event
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

    return StreamingResponse(
        stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


@router.post("/api/chat/selection")
async def chat_selection(request: SelectionRequest):
    session_id = request.session_id
    if not session_id:
        session_id = await create_session(page_context=request.chapter)

    # Save user message with selection context
    user_message = f"[Selected text: {request.selected_text[:200]}...]\n\nQuestion: {request.question}"
    await save_message(session_id, "user", user_message)

    async def stream_wrapper():
        full_response = []
        sources = []

        async for event in rag_selection_chat(
            request.selected_text,
            request.question,
            request.chapter,
            session_id,
        ):
            yield event
            if event.startswith("data: "):
                try:
                    data = json.loads(event[6:].strip())
                    if data["type"] == "token":
                        full_response.append(data["content"])
                    elif data["type"] == "sources":
                        sources = data["sources"]
                except (json.JSONDecodeError, KeyError):
                    pass

        response_text = "".join(full_response)
        await save_message(session_id, "assistant", response_text, json.dumps(sources) if sources else None)
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

    return StreamingResponse(
        stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )
