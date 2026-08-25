from dataclasses import dataclass

from app.domain.document_chunk import DocumentChunk


@dataclass
class EmbeddedDocumentChunk:
    """
    A DocumentChunk together with its embedding vector.
    """

    chunk: DocumentChunk
    vector: list[float]