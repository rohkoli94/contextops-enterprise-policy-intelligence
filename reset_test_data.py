from sqlalchemy import create_engine, text, inspect
from app.config.settings import settings
from qdrant_client import QdrantClient, models


# ============================================================
# PostgreSQL
# ============================================================

engine = create_engine(settings.database_url)

tables = inspect(engine).get_table_names(schema="public")
tables = [table for table in tables if table != "alembic_version"]

print("POSTGRES TABLES:", tables)

if tables:
    quoted_tables = ", ".join(
        f'public."{table}"'
        for table in tables
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                f"TRUNCATE TABLE {quoted_tables} "
                "RESTART IDENTITY CASCADE"
            )
        )

print("POSTGRES: ALL APPLICATION DATA DELETED")


# ============================================================
# Qdrant
# ============================================================

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)

print("QDRANT COLLECTION:", settings.qdrant_collection_name)
print("QDRANT SHARD KEY:", settings.default_tenant_id)

qdrant.delete(
    collection_name=settings.qdrant_collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[])
    ),
    shard_key_selector=models.ShardKeyWithFallback(
        target=settings.default_tenant_id,
        fallback=settings.qdrant_default_shard_key,
    ),
    wait=True,
)

print("QDRANT: ALL POINTS DELETED")

qdrant.close()

print("CLEAN RESET COMPLETE")