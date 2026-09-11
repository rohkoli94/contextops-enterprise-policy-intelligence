from dataclasses import dataclass
from typing import Any

from app.domain.document_chunk import DocumentChunk


@dataclass
class RetrievedChunk:
    """
    Represents one chunk returned by retrieval.

    The embedding vector is intentionally not returned because
    retrieval only needs the matched chunk, its metadata, and
    its relevance scores.
    """

    chunk: DocumentChunk

    # Original retrieval score returned by the vector database.
    # For hybrid retrieval this may represent the fused/RRF ranking score.
    score: float

    # Original Qdrant payload / retrieval metadata.
    metadata: dict[str, Any]

    # Score assigned by a reranker.
    #
    # This is intentionally separate from `score` because the
    # retrieval score and reranker score have different meanings.
    reranker_score: float | None = None