---
description: "Task list for Bonus Features (Auth, Personalization, Translation)"
---

# Tasks: Bonus Features

**Input**: Design documents from `/specs/003-bonus-features/`
**Prerequisites**: plan.md, spec.md, RAG chatbot backend (002) must be complete

## Phase 1: Setup

- [ ] T001 Add auth dependencies to backend/requirements.txt (passlib[bcrypt], python-jose[cryptography])
- [ ] T002 Add auth tables (users, user_profiles) to backend/app/models/database.py
- [ ] T003 Add cache tables (personalized_content, translated_content) to backend/app/models/database.py

---

## Phase 2: User Story 1 - Authentication (Priority: P1)

**Goal**: Signup/signin with background assessment

### Backend

- [ ] T004 Create backend/app/services/auth_service.py (password hashing, JWT creation/verification, user CRUD)
- [ ] T005 Create backend/app/routers/auth.py (POST /api/auth/signup, /signin, /signout, GET /me)
- [ ] T006 Add auth middleware for protected routes in backend/app/main.py
- [ ] T007 Register auth router in backend/app/main.py

### Frontend

- [ ] T008 [P] Create Book/src/components/Auth/AuthProvider.tsx (React context for auth state)
- [ ] T009 [P] Create Book/src/components/Auth/SignUpForm.tsx (with background questionnaire)
- [ ] T010 [P] Create Book/src/components/Auth/SignInForm.tsx
- [ ] T011 Create Book/src/components/Auth/UserMenu.tsx (profile display, signout button)
- [ ] T012 Update Book/src/theme/Root.tsx to wrap with AuthProvider
- [ ] T013 Add Auth button/menu to Docusaurus navbar config

**Checkpoint**: Users can sign up, sign in, and sign out

---

## Phase 3: User Story 2 - Content Personalization (Priority: P2)

**Goal**: AI-powered chapter content personalization based on user background

### Backend

- [ ] T014 Create backend/app/services/personalize_service.py (OpenAI prompt to adapt content based on user profile)
- [ ] T015 Create backend/app/routers/personalize.py (POST /api/personalize, GET /api/personalize/{chapter})
- [ ] T016 Register personalize router in main.py

### Frontend

- [ ] T017 Create Book/src/components/ChapterActions/PersonalizeButton.tsx
- [ ] T018 Create Book/src/components/ChapterActions/ChapterToolbar.tsx (combines personalize + translate buttons)
- [ ] T019 Create MDX component to embed ChapterToolbar at start of each chapter

**Checkpoint**: Logged-in users can personalize chapter content

---

## Phase 4: User Story 3 - Urdu Translation (Priority: P2)

**Goal**: AI-powered Urdu translation per chapter

### Backend

- [ ] T020 Create backend/app/services/translate_service.py (OpenAI prompt for English→Urdu translation)
- [ ] T021 Create backend/app/routers/translate.py (POST /api/translate/urdu, GET /api/translate/urdu/{chapter})
- [ ] T022 Register translate router in main.py

### Frontend

- [ ] T023 Create Book/src/components/ChapterActions/TranslateButton.tsx (with RTL CSS handling)
- [ ] T024 Add Urdu font and RTL styles to Book/src/css/custom.css

**Checkpoint**: Any user can translate chapters to Urdu

---

## Phase N: Polish

- [ ] T025 Add loading states and error handling to all auth/personalize/translate components
- [ ] T026 Test full flow: signup → personalize chapter → translate chapter

---

## Dependencies

- Phase 2 (Auth) blocks Phase 3 (Personalization needs user profile)
- Phase 4 (Translation) can run in parallel with Phase 3
- All phases require Phase 1 (Setup)
