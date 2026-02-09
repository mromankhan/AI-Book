---
description: "Task list for RAG Chatbot implementation"
---

# Tasks: RAG Chatbot for Physical AI Book

**Input**: Design documents from `/specs/002-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: Manual API testing with curl/httpie. No automated test framework required.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/` at repository root
- **Frontend widget**: `Book/src/components/ChatBot/`
- **Book content**: `Book/docs/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Backend project initialization and configuration

- [ ] T001 [P] Create backend/ directory structure per plan (app/, models/, services/, routers/, scripts/)
- [ ] T002 [P] Create backend/requirements.txt with FastAPI, uvicorn, openai, qdrant-client, asyncpg, python-dotenv, markdown dependencies
- [ ] T003 Create backend/.env.example with required environment variables (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, DATABASE_URL)
- [ ] T004 Create backend/app/config.py with environment variable loading using pydantic-settings
- [ ] T005 Create backend/app/main.py with FastAPI app, CORS middleware, and router includes

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Create backend/app/models/database.py with Neon Postgres async connection pool and session/message table creation
- [ ] T007 [P] Create backend/app/models/schemas.py with Pydantic models (ChatRequest, ChatResponse, SelectionRequest, ChunkResult)
- [ ] T008 Create backend/app/services/embeddings.py with OpenAI embedding generation (text-embedding-3-small)
- [ ] T009 Create backend/app/services/qdrant_service.py with Qdrant client init, collection creation, upsert, and search methods
- [ ] T010 Create backend/app/routers/health.py with GET /api/health endpoint
- [ ] T011 Create backend/app/services/ingestion.py with markdown parsing, chunking (512 tokens), and batch embedding+upsert pipeline
- [ ] T012 Create backend/scripts/ingest_content.py CLI script that reads Book/docs/**/*.md and runs ingestion pipeline

**Checkpoint**: Foundation ready - backend can accept connections and content is indexed

---

## Phase 3: User Story 1 - General Book Q&A (Priority: P1) MVP

**Goal**: Enable readers to ask natural language questions about book content and receive RAG-powered answers

**Independent Test**: Send POST /api/chat with a book-related question and verify response contains relevant book content

### Implementation for User Story 1

- [ ] T013 [US1] Create backend/app/services/rag_service.py with RAG pipeline: embed query → Qdrant search → build prompt with context → OpenAI chat completion with streaming
- [ ] T014 [US1] Create backend/app/routers/chat.py with POST /api/chat endpoint using StreamingResponse (SSE)
- [ ] T015 [US1] Add source references (chapter, section) to RAG responses in rag_service.py
- [ ] T016 [US1] Test general Q&A flow end-to-end: ingest content → ask question → verify accurate response

**Checkpoint**: User Story 1 should be fully functional - general Q&A works

---

## Phase 4: User Story 2 - Selected Text Q&A (Priority: P1)

**Goal**: Enable readers to ask questions about specific selected text passages

**Independent Test**: Send POST /api/chat/selection with selected text and a question, verify response is contextualized to the selection

### Implementation for User Story 2

- [ ] T017 [US2] Add POST /api/chat/selection endpoint in backend/app/routers/chat.py that accepts selected_text + question
- [ ] T018 [US2] Implement selection-context RAG in rag_service.py: use selected text as primary context, optionally augment with vector search for related content
- [ ] T019 [US2] Test selected text Q&A: send selected text + question → verify contextual response

**Checkpoint**: User Stories 1 AND 2 both work - full backend is functional

---

## Phase 5: User Story 3 - Chatbot UI Widget (Priority: P2)

**Goal**: Embed a chatbot React component in the Docusaurus book

**Independent Test**: Open any book page, click chatbot button, type a question, see streamed response

### Implementation for User Story 3

- [ ] T020 [P] [US3] Create Book/src/components/ChatBot/chatbot.module.css with floating button, chat panel, message bubble styles
- [ ] T021 [P] [US3] Create Book/src/components/ChatBot/ChatMessage.tsx component for rendering individual messages with markdown support
- [ ] T022 [US3] Create Book/src/components/ChatBot/ChatInput.tsx component with text input and send button
- [ ] T023 [US3] Create Book/src/components/ChatBot/ChatBot.tsx main component: floating button, chat panel, message list, SSE streaming integration
- [ ] T024 [US3] Create Book/src/components/ChatBot/TextSelectionPopup.tsx component that shows "Ask about this" popup on text selection
- [ ] T025 [US3] Create Book/src/theme/Root.tsx wrapper to inject ChatBot component on all pages
- [ ] T026 [US3] Test full integration: open book page → select text → ask question → see streamed response

**Checkpoint**: All user stories functional - chatbot is embedded and working in the book

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T027 [P] Create backend/Dockerfile for containerized deployment
- [ ] T028 [P] Add error handling and graceful degradation when services are unavailable
- [ ] T029 Update Book/docusaurus.config.ts to include chatbot API URL in custom fields
- [ ] T030 Create backend/README.md with setup and deployment instructions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **User Story 2 (Phase 4)**: Depends on User Story 1 (reuses rag_service.py)
- **User Story 3 (Phase 5)**: Can start after Phase 2, but needs Phase 3 for end-to-end testing
- **Polish (Phase N)**: Depends on all user stories being complete

### Parallel Opportunities

- T001 and T002 can run in parallel (different files)
- T007 can run in parallel with T006 (different files)
- T020 and T021 can run in parallel (different files)
- Frontend (Phase 5) and Backend (Phase 3-4) can be developed in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1 - General Q&A
4. **STOP and VALIDATE**: Test with curl
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Backend running, content indexed
2. User Story 1 → General Q&A works → Validate
3. User Story 2 → Selected text Q&A works → Validate
4. User Story 3 → Chatbot UI embedded in book → Full demo ready

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
