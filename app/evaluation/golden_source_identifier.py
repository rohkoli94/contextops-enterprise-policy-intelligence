from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.dependencies.rag import get_hybrid_retriever


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "datasets" / "contextops_golden.json"
OUTPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_golden_source_candidates.json"
)


class GoldenSourceIdentifier:
    """
    Generate candidate source chunks for golden-dataset examples.

    This uses the same production HybridRetriever used by
    ContextOps. The generated source IDs are candidates for
    human review; they are not automatically treated as
    ground-truth expected_source_ids.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        top_k: int | None = None,
    ) -> None:
        self.retriever = (
            retriever
            if retriever is not None
            else get_hybrid_retriever()
        )

        self.top_k = (
            top_k
            if top_k is not None
            else max(
                settings.retrieval_top_k,
                settings.rerank_candidate_limit,
            )
        )

    async def generate(
        self,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for example in examples:
            results = await self.retriever.aretrieve(
                query=example["question"],
                tenant_id=example["tenant_id"],
                top_k=self.top_k,
                filters=example.get("filters"),
            )

            sources = []

            for rank, result in enumerate(
                results,
                start=1,
            ):
                sources.append(
                    {
                        "rank": rank,
                        "chunk_id": str(
                            result.chunk.chunk_id
                        ),
                        "document_id": str(
                            result.chunk.document_id
                        ),
                        "document_version_id": str(
                            result.chunk.document_version_id
                        ),
                        "score": result.score,
                        "reranker_score": (
                            result.reranker_score
                        ),
                        "content": result.chunk.content,
                        "page_numbers": (
                            result.chunk.metadata.get(
                                "page_numbers",
                                [],
                            )
                        ),
                        "content_type": (
                            result.chunk.metadata.get(
                                "content_type"
                            )
                        ),
                    }
                )

            candidates.append(
                {
                    "question": example["question"],
                    "tenant_id": example["tenant_id"],
                    "expected_answer": example.get(
                        "expected_answer"
                    ),
                    "filters": example.get("filters"),
                    "candidate_source_ids": [
                        source["chunk_id"]
                        for source in sources
                    ],
                    "sources": sources,
                }
            )

        return candidates

    async def generate_from_file(
        self,
    ) -> list[dict[str, Any]]:
        examples = json.loads(
            DATASET_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(examples, list):
            raise ValueError(
                "Golden dataset must contain a JSON array."
            )

        return await self.generate(examples)

    def save(
        self,
        candidates: list[dict[str, Any]],
    ) -> None:
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            json.dumps(
                candidates,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


async def _main() -> None:
    identifier = GoldenSourceIdentifier()

    candidates = (
        await identifier.generate_from_file()
    )

    identifier.save(candidates)

    print()
    print("=" * 80)
    print("GOLDEN SOURCE CANDIDATES")
    print("=" * 80)
    print(f"Examples: {len(candidates)}")
    print(f"Top-K: {identifier.top_k}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print()
        print(
            f"{index:02d}. "
            f"{candidate['question']}"
        )

        source_ids = candidate[
            "candidate_source_ids"
        ]

        print(
            f"    Candidate sources: "
            f"{len(source_ids)}"
        )

        for source_id in source_ids[:5]:
            print(
                f"      - {source_id}"
            )


if __name__ == "__main__":
    asyncio.run(_main())