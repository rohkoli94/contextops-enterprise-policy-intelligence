from langchain_core.documents import Document

from app.rag.retrieval.models import RetrievedChunk


def retrieved_chunks_to_documents(
    results: list[RetrievedChunk],
) -> list[Document]:
    """
    Convert ContextOps retrieval results into LangChain Documents.
    """

    return [
        Document(
            page_content=result.chunk.content,
            metadata={
                "chunk_id": str(result.chunk.chunk_id),
                "document_id": str(result.chunk.document_id),
                "document_version_id": str(
                    result.chunk.document_version_id
                ),
                "score": result.score,
            },
        )
        for result in results
    ]