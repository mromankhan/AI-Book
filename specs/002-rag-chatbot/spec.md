# Feature Specification: RAG Chatbot for Physical AI Book

**Feature Branch**: `002-rag-chatbot`
**Created**: 2026-02-09
**Status**: Draft
**Input**: User description: "Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book using OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, and Qdrant Cloud Free Tier"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions About Book Content (Priority: P1)

As a reader of the Physical AI book, I want to ask natural language questions about any topic covered in the book and receive accurate, contextual answers sourced from the book content, so I can deepen my understanding without searching through chapters manually.

**Why this priority**: This is the core RAG chatbot functionality — the primary deliverable. Without this, the chatbot has no value.

**Independent Test**: Can be tested by asking questions like "What is ROS 2?" or "Explain bipedal locomotion" and verifying the response cites relevant book content.

**Acceptance Scenarios**:

1. **Given** a reader is on any page of the book, **When** they open the chatbot and type "What is Physical AI?", **Then** they receive an answer sourced from the book content with relevant context.
2. **Given** a reader asks a question about a specific module, **When** the chatbot processes the query, **Then** it retrieves relevant chunks from the vector store and generates an accurate answer.
3. **Given** a reader asks a question not covered in the book, **When** the chatbot processes the query, **Then** it responds honestly that the topic is not covered in the book content.

---

### User Story 2 - Ask Questions About Selected Text (Priority: P1)

As a reader, I want to select/highlight text in a chapter and ask the chatbot questions specifically about that selected text, so I can get clarification on specific passages without losing context.

**Why this priority**: This is explicitly required in the hackathon requirements — "answering questions based only on text selected by the user."

**Independent Test**: Can be tested by selecting a paragraph of text, clicking "Ask about this", and verifying the chatbot answers within the context of the selected text.

**Acceptance Scenarios**:

1. **Given** a reader selects a paragraph of text in a chapter, **When** they click "Ask about selection" and type a question, **Then** the chatbot answers based on the selected text context.
2. **Given** a reader selects code in a chapter, **When** they ask "explain this code", **Then** the chatbot explains the selected code snippet accurately.
3. **Given** no text is selected, **When** the reader uses the chatbot, **Then** it operates in general Q&A mode using full book content.

---

### User Story 3 - Chatbot UI Integration (Priority: P2)

As a reader, I want a clean, accessible chatbot interface embedded in the book that doesn't obstruct reading but is always available, so I can seamlessly switch between reading and asking questions.

**Why this priority**: Good UX is important but secondary to core functionality.

**Independent Test**: Can be tested by verifying the chatbot widget appears on all pages, opens/closes smoothly, and maintains conversation history within a session.

**Acceptance Scenarios**:

1. **Given** a reader is on any book page, **When** they look for the chatbot, **Then** they see a floating chat button in the bottom-right corner.
2. **Given** a reader opens the chatbot, **When** they type a message, **Then** the response streams in real-time and renders markdown formatting.
3. **Given** a reader closes and reopens the chatbot, **When** they look at the conversation, **Then** they see their previous messages from the current session.

---

### Edge Cases

- What happens when the Qdrant Cloud or Neon Postgres is unreachable?
- How does the chatbot handle very long selected text passages (>5000 chars)?
- What happens when the OpenAI API rate limit is hit?
- How does the system handle concurrent users querying simultaneously?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a FastAPI backend with RAG retrieval endpoints
- **FR-002**: System MUST use Qdrant Cloud for vector storage of book content embeddings
- **FR-003**: System MUST use Neon Serverless Postgres for chat session metadata and conversation history
- **FR-004**: System MUST use OpenAI API for embeddings generation and chat completion
- **FR-005**: System MUST support general Q&A about book content
- **FR-006**: System MUST support context-specific Q&A based on user-selected text
- **FR-007**: System MUST embed a chatbot UI widget in the Docusaurus book
- **FR-008**: System MUST stream responses to the user in real-time
- **FR-009**: System MUST include a content ingestion pipeline to index all book chapters into vector store
- **FR-010**: System MUST return source references (chapter/section) with each answer

### Key Entities

- **ChatMessage**: A message in a conversation (role, content, timestamp, session_id)
- **ChatSession**: A user's conversation session (session_id, created_at, page_context)
- **BookChunk**: An indexed chunk of book content (chunk_id, chapter, section, content, embedding_id)
- **VectorDocument**: A vector embedding in Qdrant (id, vector, payload with chapter/section metadata)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Chatbot returns relevant answers to book-related questions in under 5 seconds
- **SC-002**: At least 80% of answers correctly reference the relevant chapter/section
- **SC-003**: Selected text Q&A provides contextually accurate responses
- **SC-004**: All 16 chapters are indexed and searchable in the vector store
- **SC-005**: Chatbot UI is accessible on all book pages without obstructing content
- **SC-006**: System handles at least 10 concurrent users without degradation
