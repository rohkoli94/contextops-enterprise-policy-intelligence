from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    document_id: str
    source: str
    version: int
    content_hash: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)