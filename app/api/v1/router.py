from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.query.router import router as query_router

router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)