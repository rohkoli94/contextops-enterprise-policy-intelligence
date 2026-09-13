import hashlib
import math
import re
from asyncio import to_thread
from collections.abc import Awaitable, Callable

from app.config.settings import settings
from app.domain.document_chunk import DocumentChunk
from app.guardrails.base import PIIAnalyzer
from app.providers.embedding.base import (
    EmbeddingBatchRequest,
    EmbeddingRequest,
    EmbeddingProvider,
)
from app.rag.retrieval.models import RetrievedChunk
from app.rag.workflow.state import QueryState
from app.tokenization.tiktoken_counter import TiktokenCounter


DEFAULT_MMR_LAMBDA = 0.7


# ============================================================
# DEDUPLICATION
# ============================================================


def _build_deduplication_key(
    document: RetrievedChunk,
) -> str:
    """
    Build a stable deduplication key for a RetrievedChunk.

    Prefer the persisted content hash when available.

    Fall back to normalized content so that duplicate chunks
    with missing or inconsistent content hashes can still be
    detected.
    """

    chunk = document.chunk

    content_hash = str(
        getattr(
            chunk,
            "content_hash",
            "",
        )
        or ""
    ).strip()

    if content_hash:
        return f"hash:{content_hash}"

    normalized_content = (
        " ".join(
            chunk.content.split()
        )
        .strip()
        .lower()
    )

    normalized_hash = hashlib.sha256(
        normalized_content.encode(
            "utf-8"
        )
    ).hexdigest()

    return f"content:{normalized_hash}"


def _deduplicate_documents(
    documents: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Remove duplicate retrieved chunks while preserving ranking
    order.

    The first occurrence is retained because the input list is
    already ordered by retrieval/reranker relevance.
    """

    unique_documents: list[RetrievedChunk] = []
    seen_keys: set[str] = set()

    for document in documents:
        deduplication_key = (
            _build_deduplication_key(
                document
            )
        )

        if deduplication_key in seen_keys:
            continue

        seen_keys.add(
            deduplication_key
        )

        unique_documents.append(
            document
        )

    return unique_documents


# ============================================================
# VECTOR SIMILARITY
# ============================================================


def _cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns 0.0 for empty or zero-magnitude vectors.
    """

    if not left or not right:
        return 0.0

    if len(left) != len(right):
        raise ValueError(
            "Embedding dimensions do not match."
        )

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
        )
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left
        )
    )

    right_norm = math.sqrt(
        sum(
            value * value
            for value in right
        )
    )

    if (
        left_norm == 0.0
        or right_norm == 0.0
    ):
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================


def _normalize_relevance_scores(
    documents: list[RetrievedChunk],
) -> list[float]:
    """
    Normalize reranker/retrieval scores into the range [0, 1].

    Reranker scores are preferred because the input has already
    passed through the reranking stage.

    If all scores are identical, preserve equal relevance.
    """

    if not documents:
        return []

    raw_scores = [
        (
            document.reranker_score
            if document.reranker_score is not None
            else document.score
        )
        for document in documents
    ]

    minimum = min(raw_scores)
    maximum = max(raw_scores)

    if maximum == minimum:
        return [1.0] * len(
            raw_scores
        )

    return [
        (
            score - minimum
        )
        / (
            maximum - minimum
        )
        for score in raw_scores
    ]


# ============================================================
# MMR
# ============================================================

def _select_documents_with_mmr(
    *,
    documents: list[RetrievedChunk],
    embeddings: list[list[float]],
    query_embedding: list[float],
    max_documents: int,
    mmr_lambda: float,
) -> list[RetrievedChunk]:
    """
    Select a diverse subset of documents using Maximal
    Marginal Relevance (MMR).

    Standard MMR:

        MMR(candidate) =
            lambda * relevance_to_query
            - (1 - lambda) * redundancy

    relevance_to_query is calculated using cosine similarity
    between the query embedding and candidate embedding.

    redundancy is the maximum cosine similarity between the
    candidate and any already-selected document.

    Tie-breaking is diversity-first:
        1. higher MMR score
        2. lower redundancy
        3. higher normalized reranker score
        4. earlier candidate position

    This ensures that when two candidates have the same MMR
    value, a less-redundant document is preferred.
    """

    if not documents:
        return []

    if len(documents) != len(embeddings):
        raise ValueError(
            "Number of documents and embeddings must match."
        )

    if not query_embedding:
        raise ValueError(
            "Query embedding cannot be empty."
        )

    if max_documents <= 0:
        return []

    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError(
            "mmr_lambda must be between 0.0 and 1.0."
        )

    # --------------------------------------------------------
    # Query relevance
    # --------------------------------------------------------

    query_similarities = [
        _cosine_similarity(
            query_embedding,
            embedding,
        )
        for embedding in embeddings
    ]

    # --------------------------------------------------------
    # Normalized reranker scores
    #
    # Used only as a deterministic secondary tie-breaker.
    # Reranker scores must not dominate MMR.
    # --------------------------------------------------------

    normalized_relevance = (
        _normalize_relevance_scores(
            documents
        )
    )

    selected_indices: list[int] = []

    remaining_indices = set(
        range(len(documents))
    )

    target_count = min(
        max_documents,
        len(documents),
    )

    # --------------------------------------------------------
    # Greedy MMR selection
    # --------------------------------------------------------

    while (
        remaining_indices
        and len(selected_indices) < target_count
    ):
        best_index: int | None = None
        best_mmr_score = float("-inf")
        best_redundancy = float("inf")
        best_tie_break = float("-inf")

        for candidate_index in remaining_indices:
            relevance = query_similarities[
                candidate_index
            ]

            # ------------------------------------------------
            # Redundancy
            # ------------------------------------------------

            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(
                    _cosine_similarity(
                        embeddings[candidate_index],
                        embeddings[selected_index],
                    )
                    for selected_index in selected_indices
                )

            # ------------------------------------------------
            # Standard MMR
            # ------------------------------------------------

            mmr_score = (
                mmr_lambda * relevance
                - (
                    1.0 - mmr_lambda
                )
                * redundancy
            )

            # ------------------------------------------------
            # Deterministic selection
            #
            # Priority:
            #
            # 1. Higher MMR
            # 2. Lower redundancy
            # 3. Higher reranker score
            # 4. Earlier candidate
            # ------------------------------------------------

            should_select = False

            if best_index is None:
                should_select = True

            elif mmr_score > best_mmr_score:
                should_select = True

            elif math.isclose(
                mmr_score,
                best_mmr_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                if redundancy < best_redundancy:
                    should_select = True

                elif math.isclose(
                    redundancy,
                    best_redundancy,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    tie_break = normalized_relevance[
                        candidate_index
                    ]

                    if tie_break > best_tie_break:
                        should_select = True

                    elif math.isclose(
                        tie_break,
                        best_tie_break,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        if (
                            candidate_index
                            < best_index
                        ):
                            should_select = True

            if should_select:
                best_index = candidate_index
                best_mmr_score = mmr_score
                best_redundancy = redundancy
                best_tie_break = normalized_relevance[
                    candidate_index
                ]

        if best_index is None:
            break

        selected_indices.append(
            best_index
        )

        remaining_indices.remove(
            best_index
        )

    return [
        documents[index]
        for index in selected_indices
    ]


# ============================================================
# PII PROTECTION
# ============================================================


async def _redact_pii(
    *,
    text: str,
    pii_analyzer: PIIAnalyzer,
) -> tuple[str, bool, int]:
    """
    Analyze retrieved content for PII and redact detected
    entities before the content is sent to the LLM.

    Entity values are replaced with typed markers such as:

        [REDACTED_EMAIL]
        [REDACTED_PHONE]
        [REDACTED_AADHAAR]

    Redaction is performed from the end of the string toward
    the beginning so original entity offsets remain valid.
    """

    if not text:
        return (
            text,
            False,
            0,
        )

    analysis = await pii_analyzer.analyze(
        text
    )

    if not analysis.detected:
        return (
            text,
            False,
            0,
        )

    entities = sorted(
        analysis.entities,
        key=lambda entity: entity.start,
        reverse=True,
    )

    redacted_text = text

    for entity in entities:
        entity_type = (
            str(
                entity.entity_type
            )
            .strip()
            .upper()
        )

        replacement = (
            f"[REDACTED_{entity_type}]"
        )

        redacted_text = (
            redacted_text[
                : entity.start
            ]
            + replacement
            + redacted_text[
                entity.end:
            ]
        )

    return (
        redacted_text,
        True,
        len(entities),
    )


# ============================================================
# SOURCE BLOCK
# ============================================================


async def _build_source_block(
    *,
    source_index: int,
    document: RetrievedChunk,
    pii_analyzer: PIIAnalyzer,
) -> tuple[
    str,
    dict[str, object],
    bool,
    int,
]:
    """
    Build the structured source block and citation metadata.

    Retrieved document content is PII-scanned and redacted
    before the block is constructed.
    """

    chunk: DocumentChunk = (
        document.chunk
    )

    metadata = document.metadata

    document_id = str(
        metadata.get(
            "document_id",
            chunk.document_id,
        )
    )

    document_version_id = str(
        metadata.get(
            "document_version_id",
            chunk.document_version_id,
        )
    )

    chunk_id = str(
        metadata.get(
            "chunk_id",
            chunk.chunk_id,
        )
    )

    retrieval_score = (
        document.score
    )

    reranker_score = (
        document.reranker_score
    )

    (
        redacted_content,
        pii_detected,
        pii_count,
    ) = await _redact_pii(
        text=chunk.content,
        pii_analyzer=pii_analyzer,
    )

    source_block = (
        f"[SOURCE {source_index}]\n"
        f"Document ID: {document_id}\n"
        f"Document Version ID: "
        f"{document_version_id}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Retrieval Score: "
        f"{retrieval_score}\n"
        f"Reranker Score: "
        f"{reranker_score}\n"
        f"Content:\n"
        f"{redacted_content}"
    )

    citation = {
        "source": source_index,
        "document_id": document_id,
        "document_version_id": (
            document_version_id
        ),
        "chunk_id": chunk_id,
        "score": retrieval_score,
        "reranker_score": (
            reranker_score
        ),
    }

    return (
        source_block,
        citation,
        pii_detected,
        pii_count,
    )


# ============================================================
# COMPRESSION
# ============================================================


def _split_sentences(
    content: str,
) -> list[str]:
    """
    Split content into simple sentence-like units.

    This deterministic implementation intentionally avoids
    introducing another LLM call into ContextOps.
    """

    if not content.strip():
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        content.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _query_terms(
    query: str,
) -> set[str]:
    """
    Extract normalized query terms used for lightweight
    extractive relevance scoring.
    """

    return {
        token
        for token in re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            query.lower(),
        )
        if len(token) > 2
    }


def _compress_content(
    *,
    content: str,
    query: str,
    max_tokens: int,
    token_counter: TiktokenCounter,
) -> tuple[str, bool]:
    """
    Deterministically compress content when it exceeds the
    available token budget.

    Sentences are scored using query-term overlap. Higher
    relevance sentences are retained, while original document
    order is preserved in the final compressed content.

    Returns:

        (content, compressed)
    """

    if not content.strip():
        return (
            content,
            False,
        )

    if (
        token_counter.count(content)
        <= max_tokens
    ):
        return (
            content,
            False,
        )

    sentences = _split_sentences(
        content
    )

    if not sentences:
        return (
            content,
            False,
        )

    query_terms = _query_terms(
        query
    )

    if not query_terms:
        selected: list[str] = []

        for sentence in sentences:
            candidate = (
                "\n".join(
                    [
                        *selected,
                        sentence,
                    ]
                )
            )

            if (
                token_counter.count(
                    candidate
                )
                > max_tokens
            ):
                break

            selected.append(
                sentence
            )

        compressed = "\n".join(
            selected
        )

        if (
            compressed.strip()
            != content.strip()
        ):
            return (
                compressed,
                True,
            )

        return (
            content,
            False,
        )

    scored_sentences = []

    for index, sentence in enumerate(
        sentences
    ):
        sentence_terms = set(
            re.findall(
                r"\b[a-zA-Z0-9_]+\b",
                sentence.lower(),
            )
        )

        overlap = len(
            query_terms
            & sentence_terms
        )

        scored_sentences.append(
            (
                overlap,
                index,
                sentence,
            )
        )

    scored_sentences.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    selected_indices: list[int] = []
    selected_sentences: list[str] = []

    for (
        overlap,
        index,
        sentence,
    ) in scored_sentences:
        if (
            overlap == 0
            and selected_sentences
        ):
            continue

        candidate_indices = [
            *selected_indices,
            index,
        ]

        candidate_indices.sort()

        candidate = "\n".join(
            sentences[selected_index]
            for selected_index in (
                candidate_indices
            )
        )

        if (
            token_counter.count(
                candidate
            )
            <= max_tokens
        ):
            selected_indices = (
                candidate_indices
            )

            selected_sentences = [
                sentences[
                    selected_index
                ]
                for selected_index in (
                    candidate_indices
                )
            ]

    if not selected_sentences:
        token_budget_text = ""

        for sentence in sentences:
            candidate = (
                token_budget_text
                + (
                    "\n"
                    if token_budget_text
                    else ""
                )
                + sentence
            )

            if (
                token_counter.count(
                    candidate
                )
                > max_tokens
            ):
                break

            token_budget_text = (
                candidate
            )

        if token_budget_text:
            return (
                token_budget_text,
                (
                    token_budget_text.strip()
                    != content.strip()
                ),
            )

        return (
            content,
            False,
        )

    compressed = "\n".join(
        selected_sentences
    ).strip()

    if not compressed:
        return (
            content,
            False,
        )

    return (
        compressed,
        compressed != content.strip(),
    )


# ============================================================
# TOKEN-SAFE SOURCE BLOCK
# ============================================================


def _fit_source_to_budget(
    *,
    source_block: str,
    query: str,
    max_tokens: int,
    token_counter: TiktokenCounter,
) -> tuple[str, bool]:
    """
    Compress a complete source block to fit the available
    token budget.

    Metadata/header lines are preserved while content is
    compressed.
    """

    if (
        token_counter.count(source_block)
        <= max_tokens
    ):
        return (
            source_block,
            False,
        )

    marker = "Content:\n"

    if marker not in source_block:
        return (
            source_block,
            False,
        )

    header, content = (
        source_block.split(
            marker,
            maxsplit=1,
        )
    )

    header = (
        header.strip()
        + "\n"
        + marker
    )

    header_tokens = (
        token_counter.count(
            header
        )
    )

    if header_tokens >= max_tokens:
        return (
            header,
            True,
        )

    remaining_tokens = (
        max_tokens - header_tokens
    )

    compressed_content, compressed = (
        _compress_content(
            content=content,
            query=query,
            max_tokens=(
                remaining_tokens
            ),
            token_counter=token_counter,
        )
    )

    rebuilt = (
        header
        + compressed_content.strip()
    )

    if (
        token_counter.count(
            rebuilt
        )
        <= max_tokens
    ):
        return (
            rebuilt,
            compressed,
        )

    return (
        header,
        True,
    )


# ============================================================
# CONTEXTOPS NODE
# ============================================================


def create_contextops_node(
    *,
    embedding_provider: EmbeddingProvider,
    pii_analyzer: PIIAnalyzer,
    max_documents: int | None = None,
    max_tokens: int | None = None,
    token_counter: TiktokenCounter | None = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> Callable[
    [QueryState],
    Awaitable[QueryState],
]:
    """
    Create the ContextOps context-assembly node.

    Responsibilities:

        - deduplicate retrieved chunks
        - select diverse candidates using MMR
        - enforce document limits
        - enforce token budgets
        - protect retrieved context from PII
        - compress oversized source content
        - preserve source boundaries
        - preserve citation metadata
    """

    configured_max_documents = (
        max_documents
        if max_documents is not None
        else settings.context_max_documents
    )

    configured_max_tokens = (
        max_tokens
        if max_tokens is not None
        else settings.context_max_tokens
    )

    if configured_max_documents <= 0:
        raise ValueError(
            "context_max_documents must be greater than zero."
        )

    if configured_max_tokens <= 0:
        raise ValueError(
            "context_max_tokens must be greater than zero."
        )

    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError(
            "mmr_lambda must be between 0.0 and 1.0."
        )

    counter = (
        token_counter
        if token_counter is not None
        else TiktokenCounter(
            model_name=(
                settings.foundry_embedding_model_name
            ),
        )
    )

    async def node(
        state: QueryState,
    ) -> QueryState:
        documents: list[
            RetrievedChunk
        ] = state.get(
            "reranked_documents",
            [],
        )

        # ----------------------------------------------------
        # Empty retrieval
        # ----------------------------------------------------

        if not documents:
            return {
                **state,
                "context": (
                    "No relevant policy documents "
                    "were found."
                ),
                "citations": [],
                "context_token_count": 0,
                "context_pii_detected": False,
                "context_pii_entity_count": 0,
                "context_compressed": False,
                "context_compressed_document_count": 0,
            }

        # ----------------------------------------------------
        # Deduplication
        # ----------------------------------------------------

        unique_documents = (
            _deduplicate_documents(
                documents
            )
        )

        if not unique_documents:
            return {
                **state,
                "context": (
                    "No relevant policy documents "
                    "were found."
                ),
                "citations": [],
                "context_token_count": 0,
                "context_pii_detected": False,
                "context_pii_entity_count": 0,
                "context_compressed": False,
                "context_compressed_document_count": 0,
            }

        # ----------------------------------------------------
        # Retrieval query
        # ----------------------------------------------------

        retrieval_query = (
            state.get(
                "contextualized_query"
            )
            or state["query"]
        ).strip()

        if not retrieval_query:
            return {
                **state,
                "context": (
                    "No relevant policy documents "
                    "were found."
                ),
                "citations": [],
                "context_token_count": 0,
                "context_pii_detected": False,
                "context_pii_entity_count": 0,
                "context_compressed": False,
                "context_compressed_document_count": 0,
            }

        # ----------------------------------------------------
        # Candidate text
        # ----------------------------------------------------

        candidate_texts = [
            document.chunk.content
            for document in unique_documents
        ]

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding_response = (
            await embedding_provider.agenerate(
                EmbeddingRequest(
                    text=retrieval_query
                )
            )
        )

        query_embedding = (
            query_embedding_response.vector
        )

        # ----------------------------------------------------
        # Document embeddings
        # ----------------------------------------------------

        document_embedding_response = (
            await to_thread(
                embedding_provider.generate_batch,
                EmbeddingBatchRequest(
                    texts=candidate_texts
                ),
            )
        )

        document_embeddings = (
            document_embedding_response.vectors
        )

        # ----------------------------------------------------
        # Validate embedding count
        # ----------------------------------------------------

        if len(document_embeddings) != len(
            unique_documents
        ):
            raise ValueError(
                "Embedding count does not match "
                "candidate document count."
            )

        # ----------------------------------------------------
        # MMR selection
        # ----------------------------------------------------

        selected_documents = (
            _select_documents_with_mmr(
                documents=unique_documents,
                embeddings=document_embeddings,
                query_embedding=query_embedding,
                max_documents=(
                    configured_max_documents
                ),
                mmr_lambda=mmr_lambda,
            )
        )

        if not selected_documents:
            return {
                **state,
                "context": (
                    "No relevant policy documents "
                    "were found."
                ),
                "citations": [],
                "context_token_count": 0,
                "context_pii_detected": False,
                "context_pii_entity_count": 0,
                "context_compressed": False,
                "context_compressed_document_count": 0,
            }

        # ----------------------------------------------------
        # Build source blocks
        # ----------------------------------------------------

        source_blocks: list[
            tuple[
                RetrievedChunk,
                str,
                dict[str, object],
                bool,
                int,
            ]
        ] = []

        for document_index, document in enumerate(
            selected_documents,
            start=1,
        ):
            (
                source_block,
                citation,
                pii_detected,
                pii_count,
            ) = await _build_source_block(
                source_index=document_index,
                document=document,
                pii_analyzer=pii_analyzer,
            )

            source_blocks.append(
                (
                    document,
                    source_block,
                    citation,
                    pii_detected,
                    pii_count,
                )
            )

        # ----------------------------------------------------
        # Token-aware context packing
        # ----------------------------------------------------

        context_parts: list[str] = []
        citations: list[
            dict[str, object]
        ] = []

        context_token_count = 0

        context_pii_detected = False
        context_pii_entity_count = 0

        context_compressed = False
        context_compressed_document_count = 0

        for (
            _document,
            source_block,
            citation,
            pii_detected,
            pii_count,
        ) in source_blocks:
            available_tokens = (
                configured_max_tokens
                - context_token_count
            )

            if available_tokens <= 0:
                break

            # ------------------------------------------------
            # Direct fit
            # ------------------------------------------------

            source_tokens = (
                counter.count(
                    source_block
                )
            )

            final_source_block = source_block
            source_was_compressed = False

            if (
                source_tokens
                > available_tokens
            ):
                (
                    final_source_block,
                    source_was_compressed,
                ) = _fit_source_to_budget(
                    source_block=source_block,
                    query=retrieval_query,
                    max_tokens=(
                        available_tokens
                    ),
                    token_counter=counter,
                )

            final_source_tokens = (
                counter.count(
                    final_source_block
                )
            )

            if (
                final_source_tokens <= 0
                or final_source_tokens
                > available_tokens
            ):
                continue

            context_parts.append(
                final_source_block
            )

            context_token_count += (
                final_source_tokens
            )

            citations.append(
                citation
            )

            if pii_detected:
                context_pii_detected = True
                context_pii_entity_count += (
                    pii_count
                )

            if source_was_compressed:
                context_compressed = True
                context_compressed_document_count += (
                    1
                )

        # ----------------------------------------------------
        # No usable context
        # ----------------------------------------------------

        if not context_parts:
            context = (
                "No relevant policy documents "
                "were found."
            )
        else:
            context = "\n\n".join(
                context_parts
            )

        return {
            **state,
            "context": context,
            "citations": citations,
            "context_token_count": (
                context_token_count
            ),
            "context_pii_detected": (
                context_pii_detected
            ),
            "context_pii_entity_count": (
                context_pii_entity_count
            ),
            "context_compressed": (
                context_compressed
            ),
            "context_compressed_document_count": (
                context_compressed_document_count
            ),
        }

    return node