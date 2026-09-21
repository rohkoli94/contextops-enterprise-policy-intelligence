import asyncio
import hashlib
import uuid

from app.dependencies.rag import (
    get_embedding_provider,
    get_sparse_embedding_provider,
    get_vector_store,
)
from app.domain.document_chunk import DocumentChunk
from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.providers.embedding.base import (
    EmbeddingRequest,
)


CONTENT = (
    "Business loan applications require a minimum annual "
    "revenue of $500,000 for underwriting review."
)

TENANT_ID = "underwriting-test"


async def main() -> None:
    print("1. Getting providers...")

    embedding_provider = (
        get_embedding_provider()
    )

    sparse_provider = (
        get_sparse_embedding_provider()
    )

    vector_store = get_vector_store()

    try:
        print()
        print("2. Generating Microsoft Foundry embedding...")

        embedding_response = (
            await embedding_provider.agenerate(
                EmbeddingRequest(
                    text=CONTENT,
                )
            )
        )

        print(
            "   Dense dimension:",
            len(embedding_response.vector),
        )

        print(
            "   Dense model:",
            embedding_response.model,
        )

        print()
        print("3. Generating BM25 sparse embedding...")

        sparse_vector = (
            await sparse_provider.agenerate(
                CONTENT
            )
        )

        print(
            "   Sparse dimensions:",
            len(sparse_vector.indices),
        )

        document_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=version_id,
            element_ids=[
                str(uuid.uuid4())
            ],
            content=CONTENT,
            chunk_index=0,
            content_hash=hashlib.sha256(
                CONTENT.encode("utf-8")
            ).hexdigest(),
            metadata={
                "tenant_id": TENANT_ID,
                "categories": [
                    "business-loan",
                    "underwriting",
                ],
                "tags": [
                    "integration-test",
                ],
                "content_type": "text",
                "hierarchy_path": [],
                "page_numbers": [1],
            },
        )

        embedded_chunk = (
            EmbeddedDocumentChunk(
                chunk=chunk,
                vector=embedding_response.vector,
                sparse_vector=sparse_vector,
            )
        )

        print()
        print(
            "4. Upserting through "
            "ContextOps QdrantVectorStore..."
        )

        await vector_store.aupsert(
            [embedded_chunk]
        )

        print()
        print("SUCCESS")
        print("CHUNK_ID:", chunk_id)
        print("TENANT_ID:", TENANT_ID)
        print(
            "DENSE_DIMENSION:",
            len(embedding_response.vector),
        )
        print(
            "SPARSE_DIMENSIONS:",
            len(sparse_vector.indices),
        )

    finally:
        await vector_store.aclose()


if __name__ == "__main__":
    asyncio.run(main())