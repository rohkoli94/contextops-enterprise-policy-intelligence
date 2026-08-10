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

┌─────────────────┐
│    Streamlit    │
│       UI        │
│    :8501        │
└─────────────────┘
        │
        │
    NOT YET
        │
        ▼
┌─────────────────┐
│     FastAPI     │
│      API        │
│    :8000        │
└─────────────────┘
        │
        ▼
GET /api/v1/health