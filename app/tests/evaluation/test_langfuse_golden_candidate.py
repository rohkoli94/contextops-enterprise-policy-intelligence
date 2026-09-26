import json
from pathlib import Path
from typing import Any


TRACE_ID = "8416eeadc888ead049d6734e4dd6b62d"

ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "langfuse_trace_inspection.json"
)


TARGET_OBSERVATIONS = {
    "hybrid_retrieval",
    "rerank",
    "retrieval_validation",
    "route_after_retrieval_validation",
    "llm_generation",
    "grounding",
    "response",
}


def _parse_json_value(value: Any) -> Any:
    """
    Langfuse observations can contain input/output as JSON strings.

    Convert JSON strings into Python structures while leaving
    already-parsed values unchanged.
    """
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_inspection_artifact() -> list[dict[str, Any]]:
    """
    Load observations from the Langfuse inspection artifact.
    """
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Langfuse inspection artifact not found: {ARTIFACT_PATH}"
        )

    payload = json.loads(
        ARTIFACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Langfuse inspection artifact must contain "
            "a JSON object."
        )

    if payload.get("trace_id") != TRACE_ID:
        raise ValueError(
            "Langfuse inspection artifact trace_id does not "
            f"match expected trace: {TRACE_ID}"
        )

    matched_observations = payload.get(
        "matched_observations"
    )

    if not isinstance(
        matched_observations,
        dict,
    ):
        raise ValueError(
            "Langfuse inspection artifact must contain "
            "'matched_observations' as a JSON object."
        )

    observations: list[dict[str, Any]] = []

    for observation_name, matches in (
        matched_observations.items()
    ):
        if not isinstance(matches, list):
            raise ValueError(
                f"Observation '{observation_name}' must "
                "contain a list of matches."
            )

        for observation in matches:
            if not isinstance(
                observation,
                dict,
            ):
                raise ValueError(
                    f"Observation '{observation_name}' "
                    "contains a non-object match."
                )

            observations.append(observation)

    return observations


def _find_observation(
    observations: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    matches = [
        observation
        for observation in observations
        if observation.get("name") == name
    ]

    assert matches, (
        f"Required Langfuse observation not found: {name}"
    )

    assert len(matches) == 1, (
        f"Expected exactly one {name} observation, "
        f"found {len(matches)}."
    )

    return matches[0]


def _extract_state(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract the workflow state from a Langfuse observation.

    Langfuse currently stores observation input/output as
    JSON strings for these LangGraph observations.
    """
    output = _parse_json_value(
        observation.get("output")
    )

    if isinstance(output, dict):
        return output

    input_value = _parse_json_value(
        observation.get("input")
    )

    if isinstance(input_value, dict):
        return input_value

    return {}


def _extract_documents(
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
        parsed = _parse_json_value(
            observation.get(field_name)
        )

        if not isinstance(
            parsed,
            dict,
        ):
            continue

        for key in (
            "retrieved_documents",
            "reranked_documents",
        ):
            documents = parsed.get(key)

            if isinstance(
                documents,
                list,
            ):
                candidates.extend(
                    document
                    for document in documents
                    if isinstance(
                        document,
                        dict,
                    )
                )

    return candidates


def _normalize_source(
    document: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    """
    Normalize one retrieved/reranked document.
    """
    metadata = document.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    page_numbers = metadata.get(
        "page_numbers"
    )

    if page_numbers is None:
        page_numbers = document.get(
            "page_numbers"
        )

    hierarchy_path = metadata.get(
        "hierarchy_path"
    )

    if hierarchy_path is None:
        hierarchy_path = document.get(
            "hierarchy_path"
        )

    content_type = metadata.get(
        "content_type"
    )

    if content_type is None:
        content_type = document.get(
            "content_type"
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
            document.get(
                "document_version_id"
            )
            or metadata.get(
                "document_version_id"
            )
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
            else document.get(
                "retrieval_score"
            )
        ),
        "reranker_score": document.get(
            "reranker_score"
        ),
        "reranker_rank": (
            metadata.get(
                "reranker_rank"
            )
            if metadata.get(
                "reranker_rank"
            ) is not None
            else document.get(
                "reranker_rank"
            )
        ),
        "reranker_model": (
            metadata.get(
                "reranker_model"
            )
            if metadata.get(
                "reranker_model"
            ) is not None
            else document.get(
                "reranker_model"
            )
        ),
    }


def _build_sources(
    hybrid: dict[str, Any],
    rerank: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a unique source list from hybrid retrieval and reranking.

    The final source order follows the first appearance in the
    Langfuse observations.
    """
    all_documents = (
        _extract_documents(hybrid)
        + _extract_documents(rerank)
    )

    unique_documents: dict[
        str,
        dict[str, Any],
    ] = {}

    for document in all_documents:
        metadata = document.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        chunk_id = (
            document.get("chunk_id")
            or metadata.get("chunk_id")
        )

        if chunk_id:
            unique_documents[
                str(chunk_id)
            ] = document

    return [
        _normalize_source(
            document,
            rank=index,
        )
        for index, document in enumerate(
            unique_documents.values(),
            start=1,
        )
    ]


def _build_golden_candidate(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Transform one Langfuse trace into a reviewable
    golden-dataset candidate.

    IMPORTANT:
    The production answer is stored as observed_answer.
    It is never copied into expected_answer.
    expected_answer remains empty until ground truth
    has been independently verified.
    """
    hybrid = _find_observation(
        observations,
        "hybrid_retrieval",
    )

    rerank = _find_observation(
        observations,
        "rerank",
    )

    validation = _find_observation(
        observations,
        "retrieval_validation",
    )

    route = _find_observation(
        observations,
        "route_after_retrieval_validation",
    )

    llm_generation = _find_observation(
        observations,
        "llm_generation",
    )

    grounding = _find_observation(
        observations,
        "grounding",
    )

    response = _find_observation(
        observations,
        "response",
    )

    hybrid_state = _extract_state(
        hybrid
    )

    validation_state = _extract_state(
        validation
    )

    route_output = _parse_json_value(
        route.get("output")
    )

    if isinstance(
        route_output,
        dict,
    ):
        route_value = route_output.get(
            "route"
        )

        if route_value is None:
            route_value = route_output.get(
                "retrieval_validation_route"
            )
    else:
        route_value = route_output

    response_state = _extract_state(
        response
    )

    grounding_state = _extract_state(
        grounding
    )

    llm_generation_state = _extract_state(
        llm_generation
    )

    # --------------------------------------------------------
    # Core request information
    # --------------------------------------------------------

    question = (
        response_state.get("query")
        or hybrid_state.get("query")
    )

    contextualized_query = (
        response_state.get(
            "contextualized_query"
        )
        or hybrid_state.get(
            "contextualized_query"
        )
    )

    tenant_id = (
        response_state.get("tenant_id")
        or hybrid_state.get("tenant_id")
    )

    conversation_id = (
        response_state.get(
            "conversation_id"
        )
        or hybrid_state.get(
            "conversation_id"
        )
    )

    filters = (
        response_state.get("filters")
        if "filters" in response_state
        else hybrid_state.get("filters")
    )

    # --------------------------------------------------------
    # Retrieval validation
    # --------------------------------------------------------

    retrieval_sufficient = (
        validation_state.get(
            "retrieval_sufficient"
        )
    )

    retrieval_confidence = (
        validation_state.get(
            "retrieval_confidence"
        )
    )

    retrieval_score = (
        validation_state.get(
            "retrieval_score"
        )
    )

    retrieval_reason = (
        validation_state.get(
            "retrieval_reason"
        )
    )

    retrieval_signals = (
        validation_state.get(
            "retrieval_signals"
        )
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    observed_answer = response_state.get(
        "answer"
    )

    if observed_answer is None:
        observed_answer = llm_generation_state.get(
            "answer"
        )

    # --------------------------------------------------------
    # Final citations
    # --------------------------------------------------------

    citations = response_state.get(
        "citations"
    )

    if not isinstance(
        citations,
        list,
    ):
        citations = []

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    grounding_status = (
        response_state.get(
            "grounding_status"
        )
        or grounding_state.get(
            "grounding_status"
        )
    )

    grounding_reason = (
        response_state.get(
            "grounding_reason"
        )
        or grounding_state.get(
            "grounding_reason"
        )
    )

    grounding_supported_sources = (
        response_state.get(
            "grounding_supported_sources"
        )
    )

    if grounding_supported_sources is None:
        grounding_supported_sources = (
            grounding_state.get(
                "grounding_supported_sources"
            )
        )

    if not isinstance(
        grounding_supported_sources,
        list,
    ):
        grounding_supported_sources = []

    # --------------------------------------------------------
    # Final context
    # --------------------------------------------------------

    context = response_state.get(
        "context"
    )

    context_token_count = response_state.get(
        "context_token_count"
    )

    context_pii_detected = response_state.get(
        "context_pii_detected"
    )

    context_pii_entity_count = response_state.get(
        "context_pii_entity_count"
    )

    context_compressed = response_state.get(
        "context_compressed"
    )

    context_compressed_document_count = (
        response_state.get(
            "context_compressed_document_count"
        )
    )

    # --------------------------------------------------------
    # Timings
    # --------------------------------------------------------

    timings = response_state.get(
        "timings"
    )

    if not isinstance(
        timings,
        dict,
    ):
        timings = {}

    # --------------------------------------------------------
    # Source information
    # --------------------------------------------------------

    sources = _build_sources(
        hybrid,
        rerank,
    )

    # --------------------------------------------------------
    # LLM generation metadata
    # --------------------------------------------------------

    llm_generation_metadata = {
        "model": llm_generation.get(
            "model"
        ),
        "model_id": llm_generation.get(
            "modelId"
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

    # --------------------------------------------------------
    # Response metadata
    # --------------------------------------------------------

    response_metadata = response.get(
        "metadata"
    )

    if not isinstance(
        response_metadata,
        dict,
    ):
        response_metadata = {}

    # Keep only workflow/application metadata.
    # Avoid copying Langfuse SDK/resource metadata.
    final_response_metadata = {
        key: response_metadata.get(key)
        for key in (
            "conversation_id",
            "tenant_id",
            "environment",
            "application",
            "langgraph_node",
            "langgraph_step",
        )
        if key in response_metadata
    }

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    return {
        "id": TRACE_ID,
        "trace_id": TRACE_ID,
        "trace_name": hybrid.get(
            "traceName"
        ),
        "environment": (
            response.get("environment")
            or hybrid.get("environment")
        ),

        "question": question,
        "contextualized_query": (
            contextualized_query
        ),
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "filters": filters,

        # Production output.
        # NOT ground truth.
        "observed_answer": observed_answer,

        # Must remain empty until independently
        # reviewed/verified.
        "expected_answer": None,

        "retrieval": {
            "retrieval_sufficient": (
                retrieval_sufficient
            ),
            "retrieval_confidence": (
                retrieval_confidence
            ),
            "retrieval_score": retrieval_score,
            "retrieval_reason": retrieval_reason,
            "retrieval_signals": (
                retrieval_signals
            ),
        },

        "routing": {
            "retrieval_validation_route": (
                route_value
            ),
        },

        "retrieved_sources": sources,

        "citations": citations,

        "grounding": {
            "status": grounding_status,
            "reason": grounding_reason,
            "supported_sources": (
                grounding_supported_sources
            ),
        },

        "context": context,

        "context_token_count": (
            context_token_count
        ),

        "context_metadata": {
            "pii_detected": (
                context_pii_detected
            ),
            "pii_entity_count": (
                context_pii_entity_count
            ),
            "compressed": (
                context_compressed
            ),
            "compressed_document_count": (
                context_compressed_document_count
            ),
        },

        "timings": timings,

        "llm_generation": (
            llm_generation_metadata
        ),

        "response_metadata": (
            final_response_metadata
        ),

        "review": {
            "status": "pending",
            "expected_answer_verified": False,
            "expected_sources_verified": False,
        },
    }


def test_langfuse_trace_contains_required_observations() -> None:
    observations = _load_inspection_artifact()

    names = {
        observation.get("name")
        for observation in observations
    }

    missing = (
        TARGET_OBSERVATIONS - names
    )

    assert not missing, (
        f"Missing required observations: {missing}"
    )


def test_transform_langfuse_trace_into_golden_candidate() -> None:
    observations = _load_inspection_artifact()

    candidate = _build_golden_candidate(
        observations
    )

    assert candidate[
        "trace_id"
    ] == TRACE_ID

    assert candidate[
        "question"
    ] == "tell me Business Structure ?"

    assert candidate[
        "tenant_id"
    ] == "contextops"

    assert candidate[
        "conversation_id"
    ] == (
        "8a1bebec-ae0a-465e-8043-02d86e3e13a2"
    )

    assert candidate[
        "contextualized_query"
    ]

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval = candidate[
        "retrieval"
    ]

    assert retrieval[
        "retrieval_sufficient"
    ] is True

    assert retrieval[
        "retrieval_confidence"
    ] == "sufficient"

    assert retrieval[
        "retrieval_score"
    ] == 2.297222375869751

    assert retrieval[
        "retrieval_reason"
    ] == (
        "Usable reranked evidence is available, "
        "but confidence is not strongly separated."
    )

    assert retrieval[
        "retrieval_signals"
    ] == {
        "document_count": 20,
        "usable_document_count": 20,
        "reranker_score_count": 20,
        "top_reranker_score": 2.297222375869751,
        "second_reranker_score": 1.3879516124725342,
        "reranker_score_gap": 0.9092707633972168,
    }

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    assert candidate[
        "routing"
    ][
        "retrieval_validation_route"
    ] == "contextops"

    # --------------------------------------------------------
    # Retrieved sources
    # --------------------------------------------------------

    sources = candidate[
        "retrieved_sources"
    ]

    assert sources

    first_source = sources[0]

    assert first_source[
        "chunk_id"
    ]

    assert first_source[
        "document_id"
    ]

    assert first_source[
        "document_version_id"
    ]

    assert first_source[
        "content"
    ]

    # --------------------------------------------------------
    # Observed answer
    # --------------------------------------------------------

    assert candidate[
        "observed_answer"
    ]

    assert (
        "Sole Proprietorship"
        in candidate["observed_answer"]
    )

    assert (
        "Private Limited Company"
        in candidate["observed_answer"]
    )

    # --------------------------------------------------------
    # Expected answer
    # --------------------------------------------------------

    assert candidate[
        "expected_answer"
    ] is None

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    assert candidate[
        "citations"
    ]

    assert {
        citation["source"]
        for citation in candidate["citations"]
    } == {
        1,
        2,
        3,
        4,
        5,
    }

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    grounding = candidate[
        "grounding"
    ]

    assert grounding[
        "status"
    ] == "grounded"

    assert grounding[
        "reason"
    ] == (
        "Answer references available evidence sources."
    )

    assert grounding[
        "supported_sources"
    ] == [1, 2, 5]

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    assert candidate[
        "context"
    ]

    assert candidate[
        "context_token_count"
    ] == 1006

    # --------------------------------------------------------
    # Review state
    # --------------------------------------------------------

    assert candidate[
        "review"
    ][
        "status"
    ] == "pending"

    assert candidate[
        "review"
    ][
        "expected_answer_verified"
    ] is False

    assert candidate[
        "review"
    ][
        "expected_sources_verified"
    ] is False