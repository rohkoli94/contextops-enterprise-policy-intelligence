from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"


@dataclass
class DocumentElement:
    element_id: str
    document_id: str
    page_number: int
    content_type: ContentType
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)