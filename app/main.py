import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.api.v2.router import router as v2_router

from app.config.settings import settings

from app.dependencies.startup import (
    initialize_rag,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Application startup and shutdown lifecycle.

    Startup:
        Initialize RAG infrastructure once.

    Shutdown:
        No explicit cleanup is currently required.
    """

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    initialize_rag()

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    # No explicit cleanup required currently.
    pass


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ContextOps",
    description="Enterprise Policy Intelligence Platform",
    version=settings.app_version,
    lifespan=lifespan,
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(
    v1_router,
    prefix="/api/v1",
)

app.include_router(
    v2_router,
    prefix="/api/v2",
)