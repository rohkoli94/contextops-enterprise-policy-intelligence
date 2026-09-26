from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class GoldenDatasetResponse(BaseModel):
    examples: list[dict[str, Any]]