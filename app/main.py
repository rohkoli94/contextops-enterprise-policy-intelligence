import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.api.v2.router import router as v2_router
from app.config.settings import settings
from app.db.session import async_engine
from app.dependencies.startup import initialize_application


# ============================================================
# WINDOWS + ASYNC PSYCOPG COMPATIBILITY
# ============================================================
#
# psycopg 3 async connections cannot use Windows'
# default ProactorEventLoop.
#
# Production Linux deployments are unaffected.
#
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
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
        Initialize shared application infrastructure
        and services once.

    Shutdown:
        Release asynchronous clients/resources.
    """

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    initialize_application(app)

    try:
        yield

    finally:

        # ----------------------------------------------------
        # SHUTDOWN
        # ----------------------------------------------------

        # ----------------------------------------------------
        # CLOSE QUERY SERVICE ASYNC RESOURCES
        # ----------------------------------------------------

        query_service = getattr(
            app.state,
            "query_service",
            None,
        )

        if query_service is not None:

            # ------------------------------------------------
            # CLOSE LLM PROVIDER
            # ------------------------------------------------

            llm_provider = (
                query_service.llm_provider
            )

            close_method = getattr(
                llm_provider,
                "aclose",
                None,
            )

            if close_method is not None:
                await close_method()

            # ------------------------------------------------
            # CLOSE VECTOR STORE ASYNC CLIENT
            # ------------------------------------------------
            #
            # QueryService contains the shared HybridRetriever.
            # HybridRetriever directly contains the VectorStore.
            #
            # Therefore:
            #
            # QueryService
            #     ↓
            # HybridRetriever
            #     ↓
            # VectorStore
            #

            hybrid_retriever = (
                query_service.hybrid_retriever
            )

            vector_store = getattr(
                hybrid_retriever,
                "vector_store",
                None,
            )

            close_method = getattr(
                vector_store,
                "aclose",
                None,
            )

            if close_method is not None:
                await close_method()

        # ----------------------------------------------------
        # CLOSE ASYNC DATABASE ENGINE
        # ----------------------------------------------------
        #
        # Used by the asynchronous query/conversation path.
        #
        # This releases the SQLAlchemy async connection pool.
        #

        await async_engine.dispose()


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