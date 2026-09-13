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

logger = logging.getLogger(
    "contextops.lifecycle"
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
        Release explicitly registered asynchronous
        application resources and the database engine.
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
        # CLOSE REGISTERED APPLICATION RESOURCES
        # ----------------------------------------------------
        #
        # Shared providers/clients will be registered on
        # app.state by the application composition layer.
        #
        # Each resource is expected to expose:
        #
        #     aclose()
        #
        # or:
        #
        #     close()
        #
        # This keeps lifecycle management out of QueryService.
        #

        resources = getattr(
            app.state,
            "shutdown_resources",
            [],
        )

        for resource in resources:
            if resource is None:
                continue

            try:
                close_method = getattr(
                    resource,
                    "aclose",
                    None,
                )

                if close_method is not None:
                    await close_method()
                    continue

                close_method = getattr(
                    resource,
                    "close",
                    None,
                )

                if close_method is not None:
                    result = close_method()

                    if asyncio.iscoroutine(
                        result
                    ):
                        await result

            except Exception:
                logger.exception(
                    "Failed to close application resource: %r",
                    resource,
                )

        # ----------------------------------------------------
        # CLOSE ASYNC DATABASE ENGINE
        # ----------------------------------------------------
        #
        # This releases the SQLAlchemy async connection pool.
        #

        try:
            await async_engine.dispose()
        except Exception:
            logger.exception(
                "Failed to dispose async database engine."
            )


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