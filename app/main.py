from fastapi import FastAPI
from app.api.v1.router import router as v1_router
from app.api.v2.router import router as v2_router
from app.config.settings import settings

app = FastAPI(
    title="ContextOps",
    description="Enterprise Policy Intelligence Platform",
    version=settings.app_version
)

app.include_router(
    v1_router,
    prefix="/api/v1",
)

app.include_router(
    v2_router,
    prefix="/api/v2",
)