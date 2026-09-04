import hashlib
import json
from typing import Any


def build_query_cache_key(
    *,
    tenant_id: str,
    query: str,
    filters: dict[str, Any] | None,
    knowledge_base_version: str = "current",
    prompt_version: str = "v1",
    model_version: str = "current",
) -> str:
    """
    Build a deterministic, version-aware cache key.

    Day 18:
        knowledge_base_version/prompt/model values are supported
        structurally.

    Day 21:
        these values will be populated from the real KB/version
        and cache configuration.
    """

    payload = {
        "tenant_id": tenant_id,
        "query": query.strip(),
        "filters": filters or {},
        "knowledge_base_version": knowledge_base_version,
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