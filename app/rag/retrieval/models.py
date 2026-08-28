from dataclasses import dataclass
from typing import Any

from app.domain.document_chunk import DocumentChunk


@dataclass
class RetrievedChunk:
    """
    Represents one chunk returned by retrieval.

    The embedding vector is intentionally not returned because
    retrieval only needs the matched chunk, its metadata, and
    its relevance score.
    """

    chunk: DocumentChunk

    # Similarity score returned by the vector database.
    score: float

    # Original Qdrant payload / retrieval metadata.
    metadata: dict[str, Any]