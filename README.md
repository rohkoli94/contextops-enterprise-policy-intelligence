# ContextOps — Enterprise Policy Intelligence Platform

ContextOps is an enterprise policy intelligence platform designed to provide
reliable, grounded answers over large and complex enterprise policy documents.

The system will evolve into a multimodal, evaluation-driven RAG platform
supporting documents containing text, tables, images and diagrams.

## Day 1

Day 1 establishes the project foundation:

- Python environment managed with uv
- FastAPI backend
- Streamlit frontend
- Versioned API structure
- Basic project structure
- Local development environment

## Current Architecture

```text
Streamlit UI : 8501
        │
        ▼
FastAPI : 8000
        │
        ▼
GET /api/v1/health
```

## Day 2 — Configuration & Environment Management

Day 2 introduces centralized application configuration and environment
management.

### What was added

- Pydantic Settings for typed application configuration
- Environment variable support
- Local `.env` support for development
- `.env.example` for documenting required configuration
- FastAPI application metadata loaded from configuration
- `.env` excluded from Git

### Configuration Flow

```text
.env / Environment Variables
            │
            ▼
    Pydantic Settings
            │
            ▼
       Application
            │
            ▼
         FastAPI
```

## Day 3 — Versioned Query API & Logging

Added the first versioned query API for ContextOps.

### Added

- `POST /api/v1/query`
- Feature-based API structure
- Separate request and response schemas
- Pydantic request validation
- Swagger/OpenAPI documentation
- Centralized logging
- Logging abstraction with `get_logger()`
- Sensitive query content is not logged

### API Flow

```text
Streamlit UI
  ↓
FastAPI
  ↓
POST /api/v1/query
  ↓
Request Validation
  ↓
query_policy()
  ↓
QueryResponse
```

### Current Endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`


## Day 4 — Streamlit UI Integration

Added a Streamlit interface and connected it to the ContextOps FastAPI backend.

### Added

- Streamlit query interface
- User input validation
- FastAPI integration using HTTP
- API error handling
- Configurable backend API URL

### Flow

```text
Streamlit UI
    ↓
FastAPI Query API
    ↓
QueryResponse
    ↓
Display Answer
```

### Status

- Day 1 — Project Foundation ✅
- Day 2 — Configuration & Environment Management ✅
- Day 3 — Versioned Query API & Logging ✅
- Day 4 — Streamlit UI Integration ✅