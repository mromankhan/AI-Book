# Implementation Plan: Bonus Features (Auth, Personalization, Translation)

**Branch**: `003-bonus-features` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-bonus-features/spec.md`

## Summary

Implement three bonus features for the Physical AI book: (1) better-auth authentication with background assessment, (2) AI-powered content personalization per chapter, and (3) AI-powered Urdu translation per chapter. All features integrate with the existing Docusaurus frontend and FastAPI backend, sharing the Neon Postgres database.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript/React (frontend)
**Primary Dependencies**: better-auth (JS), FastAPI, OpenAI API, Neon Postgres
**Storage**: Neon Serverless Postgres (users, profiles, cached content)
**Target Platform**: Web (Docusaurus + FastAPI)
**Constraints**: better-auth runs in JS/Node.js; needs integration approach with Docusaurus

## Architecture

### Authentication (better-auth)

better-auth is a JS library. We'll integrate it as:
- Backend auth endpoints added to FastAPI (signup, signin, signout, session check)
- Password hashing with bcrypt via passlib
- JWT tokens for session management
- Frontend auth components in Docusaurus (SignIn/SignUp forms, AuthProvider context)

### Personalization & Translation

Both use the same pattern:
1. User clicks button on chapter page
2. Frontend sends chapter content + user profile to backend
3. Backend calls OpenAI to personalize/translate
4. Result is cached in Postgres
5. Cached result served on subsequent requests

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/signup` | Create account with background |
| POST | `/api/auth/signin` | Sign in |
| POST | `/api/auth/signout` | Sign out |
| GET | `/api/auth/me` | Get current user profile |
| POST | `/api/personalize` | Personalize chapter for user |
| GET | `/api/personalize/{chapter}` | Get cached personalization |
| POST | `/api/translate/urdu` | Translate chapter to Urdu |
| GET | `/api/translate/urdu/{chapter}` | Get cached translation |

## Project Structure

### Source Code additions

```text
backend/app/
├── routers/
│   ├── auth.py             # Auth endpoints (signup/signin/signout/me)
│   ├── personalize.py      # Personalization endpoints
│   └── translate.py        # Translation endpoints
├── services/
│   ├── auth_service.py     # Auth logic (hashing, JWT, user CRUD)
│   ├── personalize_service.py  # AI personalization logic
│   └── translate_service.py    # AI translation logic
└── models/
    └── database.py         # Add users, profiles, cache tables

Book/src/
├── components/
│   ├── Auth/
│   │   ├── AuthProvider.tsx    # Auth context provider
│   │   ├── SignInForm.tsx      # Sign in form
│   │   ├── SignUpForm.tsx      # Sign up form with background
│   │   └── UserMenu.tsx       # User menu (profile, signout)
│   ├── ChapterActions/
│   │   ├── PersonalizeButton.tsx   # Personalize button
│   │   ├── TranslateButton.tsx     # Translate to Urdu button
│   │   └── ChapterToolbar.tsx      # Combined toolbar
│   └── ...
├── theme/
│   └── Root.tsx            # Updated with AuthProvider
└── ...
```
