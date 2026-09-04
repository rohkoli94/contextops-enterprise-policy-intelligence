from collections.abc import Awaitable, Callable

from app.config.settings import settings
from app.rag.workflow.state import QueryState


def create_contextops_node(
    max_documents: int | None = None,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the ContextOps context-assembly node.

    Day 18 responsibilities:
        - select a bounded number of reranked documents
        - build structured LLM context
        - preserve citation metadata

    Day 20 responsibilities:
        - deduplication
        - MMR/diversity
        - PII protection for retrieved context
        - token-aware packing
        - context compression
    """

    configured_max_documents = (
        max_documents
        if max_documents is not None
        else settings.context_max_documents
    )

    if configured_max_documents <= 0:
        raise ValueError(
            "context_max_documents must be greater than zero."
        )

    async def node(
        state: QueryState,
    ) -> QueryState:
        documents = state.get(
            "reranked_documents",
            [],
        )

        # --------------------------------------------------
        # Context document upper bound
        # --------------------------------------------------

        selected_documents = documents[
            :configured_max_documents
        ]

        context_parts: list[str] = []
        citations: list[dict[str, object]] = []

        for index, document in enumerate(
            selected_documents,
            start=1,
        ):
            metadata = document.metadata

            document_id = str(
                metadata.get(
                    "document_id",
                    "",
                )
            )

            document_version_id = str(
                metadata.get(
                    "document_version_id",
                    "",
                )
            )

            chunk_id = str(
                metadata.get(
                    "chunk_id",
                    "",
                )
            )

            retrieval_score = metadata.get(
                "score"
            )

            reranker_score = metadata.get(
                "reranker_score"
            )

            # --------------------------------------------------
            # LLM context representation
            # --------------------------------------------------

            context_parts.append(
                (
                    f"[SOURCE {index}]\n"
                    f"Document ID: {document_id}\n"
                    f"Document Version ID: "
                    f"{document_version_id}\n"
                    f"Chunk ID: {chunk_id}\n"
                    f"Retrieval Score: "
                    f"{retrieval_score}\n"
                    f"Reranker Score: "
                    f"{reranker_score}\n"
                    f"Content:\n"
                    f"{document.page_content}"
                )
            )

            # --------------------------------------------------
            # API/UI citation representation
            # --------------------------------------------------

            citations.append(
                {
                    "source": index,
                    "document_id": document_id,
                    "document_version_id": (
                        document_version_id
                    ),
                    "chunk_id": chunk_id,
                    "score": retrieval_score,
                    "reranker_score": reranker_score,
                }
            )

        # --------------------------------------------------
        # No usable context
        # --------------------------------------------------

        if not context_parts:
            context = (
                "No relevant policy documents were found."
            )
        else:
            context = "\n\n".join(
                context_parts
            )

        return {
            **state,
            "context": context,
            "citations": citations,
        }

    return node