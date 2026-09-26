import json
from typing import Any


TARGET_OBSERVATIONS = {
    "hybrid_retrieval",
    "rerank",
    "retrieval_validation",
    "route_after_retrieval_validation",
    "llm_generation",
    "grounding",
    "response",
}


def parse_json_value(value: Any) -> Any:
    """
    Parse a Langfuse observation input/output value.

    Langfuse may return structured values directly or as
    JSON-encoded strings.
    """

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def find_observation(
    observations: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    """
    Find exactly one observation with the requested name.
    """

    matches = [
        observation
        for observation in observations
        if observation.get("name") == name
    ]

    if not matches:
        raise ValueError(
            f"Required Langfuse observation not found: {name}"
        )

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{name}' observation, "
            f"found {len(matches)}."
        )

    return matches[0]


def extract_state(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract a workflow state dictionary from an observation.

    Output is preferred because LangGraph observations generally
    contain the resulting state there.
    """

    for field_name in (
        "output",
        "input",
    ):
        parsed = parse_json_value(
            observation.get(field_name)
        )

        if isinstance(parsed, dict):
            return parsed

    return {}


def extract_documents(
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract retrieved/reranked documents from an observation.
    """

    candidates: list[dict[str, Any]] = []

    for field_name in (
        "output",
        "input",
    ):
        parsed = parse_json_value(
            observation.get(field_name)
        )

        if not isinstance(parsed, dict):
            continue

        for key in (
            "retrieved_documents",
            "reranked_documents",
        ):
            documents = parsed.get(key)

            if isinstance(documents, list):
                candidates.extend(
                    document
                    for document in documents
                    if isinstance(document, dict)
                )

    return candidates


def normalize_source(
    document: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    """
    Normalize one retrieved/reranked document.
    """

    metadata = document.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    page_numbers = (
        metadata.get("page_numbers")
        if metadata.get("page_numbers") is not None
        else document.get("page_numbers")
    )

    hierarchy_path = (
        metadata.get("hierarchy_path")
        if metadata.get("hierarchy_path") is not None
        else document.get("hierarchy_path")
    )

    content_type = (
        metadata.get("content_type")
        if metadata.get("content_type") is not None
        else document.get("content_type")
    )

    return {
        "rank": rank,
        "chunk_id": (
            document.get("chunk_id")
            or metadata.get("chunk_id")
        ),
        "document_id": (
            document.get("document_id")
            or metadata.get("document_id")
        ),
        "document_version_id": (
            document.get("document_version_id")
            or metadata.get("document_version_id")
        ),
        "content": document.get(
            "content",
            metadata.get("content"),
        ),
        "content_type": content_type,
        "page_numbers": page_numbers,
        "hierarchy_path": hierarchy_path,
        "element_ids": (
            document.get("element_ids")
            or metadata.get("element_ids")
        ),
        "content_hash": (
            document.get("content_hash")
            or metadata.get("content_hash")
        ),
        "retrieval_score": (
            document.get("score")
            if document.get("score") is not None
            else document.get("retrieval_score")
        ),
        "reranker_score": document.get(
            "reranker_score"
        ),
        "reranker_rank": (
            metadata.get("reranker_rank")
            if metadata.get("reranker_rank") is not None
            else document.get("reranker_rank")
        ),
        "reranker_model": (
            metadata.get("reranker_model")
            if metadata.get("reranker_model") is not None
            else document.get("reranker_model")
        ),
    }


def build_golden_candidate(
    trace_id: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Transform one Langfuse trace into a reviewable
    golden-dataset candidate.

    This does NOT establish ground truth.

    expected_answer and expected_source_ids remain empty
    until the candidate is reviewed.
    """

    required = TARGET_OBSERVATIONS - {
        observation.get("name")
        for observation in observations
    }

    if required:
        raise ValueError(
            "Missing required Langfuse observations: "
            + ", ".join(sorted(required))
        )

    hybrid = find_observation(
        observations,
        "hybrid_retrieval",
    )

    rerank = find_observation(
        observations,
        "rerank",
    )

    validation = find_observation(
        observations,
        "retrieval_validation",
    )

    route = find_observation(
        observations,
        "route_after_retrieval_validation",
    )

    llm_generation = find_observation(
        observations,
        "llm_generation",
    )

    grounding = find_observation(
        observations,
        "grounding",
    )

    response = find_observation(
        observations,
        "response",
    )

    # ========================================================
    # STATES
    # ========================================================

    hybrid_state = extract_state(hybrid)
    validation_state = extract_state(validation)
    route_state = extract_state(route)
    llm_generation_state = extract_state(
        llm_generation
    )
    grounding_state = extract_state(grounding)
    response_state = extract_state(response)

    # ========================================================
    # FINAL RESPONSE STATE
    # ========================================================

    # The response observation contains the consolidated
    # LangGraph final state. Prefer it for answer/citations/
    # grounding/context information.

    final_state = response_state or grounding_state

    # ========================================================
    # REQUEST INFORMATION
    # ========================================================

    question = (
        final_state.get("query")
        or hybrid_state.get("query")
    )

    contextualized_query = (
        final_state.get("contextualized_query")
        or hybrid_state.get("contextualized_query")
    )

    tenant_id = (
        final_state.get("tenant_id")
        or hybrid_state.get("tenant_id")
    )

    conversation_id = (
        final_state.get("conversation_id")
        or hybrid_state.get("conversation_id")
    )

    filters = (
        final_state.get("filters")
        or hybrid_state.get("filters")
    )

    # ========================================================
    # OBSERVED ANSWER
    # ========================================================

    observed_answer = (
        final_state.get("answer")
        or llm_generation_state.get("answer")
    )

    # ========================================================
    # RETRIEVAL VALIDATION
    # ========================================================

    retrieval_sufficient = validation_state.get(
        "retrieval_sufficient"
    )

    retrieval_confidence = validation_state.get(
        "retrieval_confidence"
    )

    retrieval_score = validation_state.get(
        "retrieval_score"
    )

    retrieval_reason = validation_state.get(
        "retrieval_reason"
    )

    retrieval_signals = validation_state.get(
        "retrieval_signals"
    )

    # ========================================================
    # SOURCES
    # ========================================================

    all_documents = (
        extract_documents(hybrid)
        + extract_documents(rerank)
    )

    unique_documents: dict[str, dict[str, Any]] = {}

    for document in all_documents:
        metadata = document.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = (
            document.get("chunk_id")
            or metadata.get("chunk_id")
        )

        if chunk_id:
            unique_documents[str(chunk_id)] = document

    retrieved_sources = [
        normalize_source(
            document=document,
            rank=index,
        )
        for index, document in enumerate(
            unique_documents.values(),
            start=1,
        )
    ]

    # ========================================================
    # CITATIONS
    # ========================================================

    citations = final_state.get(
        "citations",
        [],
    )

    if not isinstance(citations, list):
        citations = []

    # ========================================================
    # GROUNDING
    # ========================================================

    grounding_status = final_state.get(
        "grounding_status"
    )

    grounding_reason = final_state.get(
        "grounding_reason"
    )

    grounding_supported_sources = final_state.get(
        "grounding_supported_sources",
        [],
    )

    if grounding_status is None:
        grounding_status = grounding_state.get(
            "grounding_status"
        )

    if grounding_reason is None:
        grounding_reason = grounding_state.get(
            "grounding_reason"
        )

    if not grounding_supported_sources:
        grounding_supported_sources = (
            grounding_state.get(
                "grounding_supported_sources",
                [],
            )
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    context = final_state.get(
        "context"
    )

    context_token_count = final_state.get(
        "context_token_count"
    )

    context_pii_detected = final_state.get(
        "context_pii_detected"
    )

    context_pii_entity_count = final_state.get(
        "context_pii_entity_count"
    )

    context_compressed = final_state.get(
        "context_compressed"
    )

    # ========================================================
    # TIMINGS
    # ========================================================

    timings = final_state.get(
        "timings",
        {},
    )

    if not isinstance(timings, dict):
        timings = {}

    # ========================================================
    # LLM METADATA
    # ========================================================

    llm_metadata = {
        "model": (
            llm_generation.get("model")
            or llm_generation.get("modelId")
        ),
        "latency": llm_generation.get(
            "latency"
        ),
        "input_usage": llm_generation.get(
            "inputUsage"
        ),
        "output_usage": llm_generation.get(
            "outputUsage"
        ),
        "total_usage": llm_generation.get(
            "totalUsage"
        ),
        "total_cost": llm_generation.get(
            "totalCost"
        ),
    }

    # ========================================================
    # OBSERVATION METADATA
    # ========================================================

    response_metadata = response.get(
        "metadata"
    )

    if not isinstance(response_metadata, dict):
        response_metadata = {}

    # ========================================================
    # CANDIDATE
    # ========================================================

    return {
        "id": trace_id,
        "trace_id": trace_id,
        "trace_name": (
            response.get("traceName")
            or hybrid.get("traceName")
        ),
        "environment": (
            response.get("environment")
            or hybrid.get("environment")
        ),
        "question": question,
        "contextualized_query": contextualized_query,
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "filters": filters,

        "observed_answer": observed_answer,

        # Ground truth must NOT be copied from the
        # observed answer automatically.
        "expected_answer": None,
        "expected_source_ids": [],

        "retrieval": {
            "retrieval_sufficient": (
                retrieval_sufficient
            ),
            "retrieval_confidence": (
                retrieval_confidence
            ),
            "retrieval_score": retrieval_score,
            "retrieval_reason": retrieval_reason,
            "retrieval_signals": retrieval_signals,
        },

        "routing": {
            "retrieval_validation_route": (
                parse_json_value(
                    route.get("output")
                )
            ),
        },

        "retrieved_sources": retrieved_sources,

        "citations": citations,

        "grounding": {
            "status": grounding_status,
            "reason": grounding_reason,
            "supported_sources": (
                grounding_supported_sources
            ),
        },

        "context": {
            "content": context,
            "token_count": context_token_count,
            "pii_detected": context_pii_detected,
            "pii_entity_count": (
                context_pii_entity_count
            ),
            "compressed": context_compressed,
        },

        "timings": timings,

        "llm_generation": llm_metadata,

        "response_metadata": {
            "conversation_id": response_metadata.get(
                "conversation_id"
            ),
            "tenant_id": response_metadata.get(
                "tenant_id"
            ),
            "environment": response_metadata.get(
                "environment"
            ),
            "application": response_metadata.get(
                "application"
            ),
            "langgraph_node": response_metadata.get(
                "langgraph_node"
            ),
            "langgraph_step": response_metadata.get(
                "langgraph_step"
            ),
        },

        "review": {
            "status": "pending",
            "expected_answer_verified": False,
            "expected_sources_verified": False,
        },
    }