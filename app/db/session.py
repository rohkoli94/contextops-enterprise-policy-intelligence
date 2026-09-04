from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import settings


# ============================================================
# SYNCHRONOUS DATABASE ACCESS
# ============================================================
#
# Used by the existing document/ingestion path.
#
# Example:
#   DocumentService
#   DocumentIngestionService
#
engine = create_engine(
    settings.database_url,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Provide a synchronous SQLAlchemy session.

    Existing ingestion/document workflows continue using this
    session pattern.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ============================================================
# ASYNCHRONOUS DATABASE ACCESS
# ============================================================
#
# Used by the async query path / LangGraph workflow.
#
# IMPORTANT:
# Never use the synchronous SessionLocal from an async query
# node because blocking database calls can block the FastAPI
# event loop under load.
#
# psycopg 3 provides the async PostgreSQL driver/dialect.
#

ASYNC_DATABASE_URL = settings.database_url.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an asynchronous SQLAlchemy session.

    Used by the query/conversation workflow.
    """

    async with AsyncSessionLocal() as db:
        yield db