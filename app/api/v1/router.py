from fastapi import APIRouter

from app.api.v1.documents.router import (
    router as documents_router,
)
from app.api.v1.health import (
    router as health_router,
)
from app.api.v1.query.router import (
    router as query_router,
)


router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

router.include_router(
    health_router,
)


# ============================================================
# DOCUMENTS
# ============================================================

router.include_router(
    documents_router,
)


# ============================================================
# QUERY
# ============================================================

router.include_router(
    query_router,
)