from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SkippedGoldenTrace(BaseModel):
    trace_id: str
    reason: str


class GenerateGoldenFilesResponse(BaseModel):
    golden_dataset_file: str
    golden_source_candidates_file: str
    example_count: int
    candidate_count: int

    golden_dataset: list[dict[str, Any]]
    source_candidates: list[dict[str, Any]]

    processed_trace_ids: list[str]
    skipped_traces: list[SkippedGoldenTrace]
    skipped_count: int
