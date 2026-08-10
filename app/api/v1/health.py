from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health - V1"],
)


@router.get("")
def health_check():
    return {
        "status": "healthy",
        "service": "contextops",
        "api_version": "v1",
    }