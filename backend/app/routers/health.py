from datetime import datetime
from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    qdrant_status = "unknown"
    db_status = "unknown"

    # Check Qdrant
    try:
        from app.services.qdrant_service import client as qdrant_client
        qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"

    # Check Database
    try:
        from app.models.database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    status = "healthy" if qdrant_status == "connected" and db_status == "connected" else "degraded"

    return HealthResponse(
        status=status,
        qdrant=qdrant_status,
        database=db_status,
        timestamp=datetime.utcnow(),
    )
