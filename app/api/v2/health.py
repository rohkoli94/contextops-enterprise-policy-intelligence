from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health - V2"]
)


@router.get("")
def health_check():
    return {
        "status": "healthy",
        "service": "contextops",
        "api_version": "v2",
        "api_contract": "new"
    }