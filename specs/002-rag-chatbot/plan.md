# Implementation Plan: RAG Chatbot for Physical AI Book

**Branch**: `002-rag-chatbot` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-chatbot/spec.md`

## Summary

Build a RAG (Retrieval-Augmented Generation) chatbot embedded in the Physical AI Docusaurus book. The system consists of a FastAPI backend that indexes book content into Qdrant Cloud vector store, uses Neon Serverless Postgres for session/metadata storage, and OpenAI for embeddings + chat completion. A React-based chatbot widget is embedded in Docusaurus for the frontend. Supports both general Q&A and selected-text-based questions.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript/React (frontend widget)
**Primary Dependencies**: FastAPI, openai SDK, qdrant-client, psycopg2/asyncpg, Docusaurus React components
**Storage**: Qdrant Cloud (vectors), Neon Serverless Postgres (sessions/metadata)
**Testing**: Manual API testing, curl/httpie for endpoints
**Target Platform**: Web - FastAPI deployed as API, Docusaurus on GitHub Pages
**Project Type**: Web application (backend API + frontend widget)
**Performance Goals**: <5s response time, streaming support, 10 concurrent users
**Constraints**: Qdrant Cloud Free Tier limits, OpenAI API costs, CORS for cross-origin requests
**Scale/Scope**: 16 chapters indexed, ~500-1000 chunks, single chatbot instance

## Constitution Check

1. **Practical Hands-On Learning**: RAG chatbot enhances learning by providing interactive Q&A
2. **Docusaurus Standard**: Widget integrates as a React component within Docusaurus
3. **Git Workflow**: Developed in feature branch `002-rag-chatbot`
4. **Beginner-Friendly**: Chatbot makes content more accessible to beginners

All gates passed.

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file
├── spec.md              # Feature specification
├── contracts/
│   └── api.yaml         # API contract
└── tasks.md             # Task list
```

### Source Code (repository root)

```text
backend/                         # FastAPI backend
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Environment configuration
│   ├── models/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── database.py         # Neon Postgres connection & models
│   ├── services/
│   │   ├── embeddings.py       # OpenAI embeddings service
│   │   ├── qdrant_service.py   # Qdrant vector store operations
│   │   ├── rag_service.py      # RAG retrieval + generation pipeline
│   │   └── ingestion.py        # Book content ingestion pipeline
│   └── routers/
│       ├── chat.py             # Chat endpoints
│       └── health.py           # Health check endpoint
├── scripts/
│   └── ingest_content.py       # CLI script to ingest book content
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── Dockerfile                  # Container deployment

Book/src/components/
├── ChatBot/
│   ├── ChatBot.tsx             # Main chatbot widget component
│   ├── ChatMessage.tsx         # Individual message component
│   ├── ChatInput.tsx           # Message input component
│   ├── TextSelectionPopup.tsx  # Selected text action popup
│   └── chatbot.module.css      # Chatbot styles
└── ...
```

**Structure Decision**: Web application pattern with separate backend (FastAPI) and frontend (Docusaurus React component). Backend handles all AI/data operations; frontend is a lightweight chat widget.

## Architecture

### Data Flow

```
User Question → Docusaurus Widget → FastAPI Backend
                                       ├─→ OpenAI Embeddings (query → vector)
                                       ├─→ Qdrant (vector similarity search → relevant chunks)
                                       ├─→ OpenAI Chat (context + question → answer)
                                       └─→ Neon Postgres (log session/messages)
                                    ← Streamed Response ←
```

### Content Ingestion Pipeline

```
Book/docs/*.md → Parse Markdown → Chunk (512 tokens) → OpenAI Embed → Qdrant Store
                                                                    → Neon Postgres (metadata)
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send message, get RAG response (streaming) |
| POST | `/api/chat/selection` | Q&A about selected text |
| GET | `/api/health` | Health check |
| POST | `/api/ingest` | Trigger content ingestion (admin) |

### Key Decisions

1. **Streaming via SSE**: Use Server-Sent Events for real-time response streaming
2. **Chunk size 512 tokens**: Balance between context richness and retrieval precision
3. **OpenAI text-embedding-3-small**: Cost-effective embedding model (1536 dimensions)
4. **Qdrant Cloud Free Tier**: 1GB storage, sufficient for ~1000 chunks
5. **Neon Free Tier**: 0.5GB storage, sufficient for sessions and metadata

## Complexity Tracking

*No complexities or violations to track as the implementation aligns with all constitutional requirements.*
