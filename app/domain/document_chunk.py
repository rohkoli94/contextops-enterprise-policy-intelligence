from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    Represents a retrieval-ready unit created from one or more
    extracted document elements.
    """

    # Unique identifier for this chunk.
    chunk_id: str

    # Logical document this chunk belongs to.
    document_id: str

    # Specific document version this chunk belongs to.
    document_version_id: str

    # Original DocumentElement IDs that contributed to this chunk.
    #
    # One chunk can contain multiple elements, and one element
    # can also be split into multiple chunks.
    element_ids: list[str]

    # Final content that will later be embedded.
    content: str

    # Sequential position within the document version.
    chunk_index: int

    # SHA-256 hash of the final chunk content.
    # Useful for deduplication/change detection.
    content_hash: str

    # Additional retrieval/citation metadata.
    metadata: dict[str, Any] = field(default_factory=dict)