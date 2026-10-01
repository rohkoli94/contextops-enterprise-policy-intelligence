import hashlib
import json
from typing import Any


def build_query_cache_key(
    *,
    tenant_id: str,
    query: str,
    filters: dict[str, Any] | None,
    knowledge_base_version: str = "0",
    prompt_version: str = "v1",
    model_version: str = "current",
) -> str:
    """
    Build a deterministic cache key that includes all inputs that
    can change the generated answer.

    knowledge_base_version is the tenant-scoped Redis generation.
    Any successful knowledge-base update moves queries to a new
    cache namespace without scanning or deleting old entries.
    """

    payload = {
        "tenant_id": tenant_id.strip(),
        "query": query.strip(),
        "filters": filters or {},
        "knowledge_base_version": str(knowledge_base_version),
        "prompt_version": prompt_version,
        "model_version": model_version,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return f"contextops:query:{digest}"
