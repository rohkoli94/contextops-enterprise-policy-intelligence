from asyncio import to_thread

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config.settings import settings
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.reranker import Reranker


class FastEmbedReranker(Reranker):
    """
    Production cross-encoder reranker using FastEmbed/ONNX.

    The reranker:
    - receives candidates from hybrid/RRF retrieval
    - reranks at most the configured candidate limit
    - preserves the original retrieval score
    - stores the cross-encoder score separately
    - preserves retrieval metadata
    """

    def __init__(
        self,
        *,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = (
            cache_dir or settings.fastembed_cache_dir
        )
        self.threads = threads

        self._model = TextCrossEncoder(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            threads=self.threads,
            providers=["CPUExecutionProvider"],
        )

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        candidate_limit = min(
            len(candidates),
            settings.rerank_candidate_limit,
        )

        candidates_to_rerank = candidates[:candidate_limit]

        documents = [
            candidate.chunk.content
            for candidate in candidates_to_rerank
        ]

        scores = await to_thread(
            self._score,
            query.strip(),
            documents,
        )

        scored_candidates = list(
            zip(
                candidates_to_rerank,
                scores,
            )
        )

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_candidates: list[RetrievedChunk] = []

        for rank, (candidate, score) in enumerate(
            scored_candidates,
            start=1,
        ):
            reranker_score = float(score)

            metadata = dict(candidate.metadata)
            metadata["reranker_applied"] = True
            metadata["reranker_model"] = self.model_name
            metadata["reranker_rank"] = rank
            metadata["reranker_score"] = reranker_score

            reranked_candidates.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    metadata=metadata,
                    reranker_score=reranker_score,
                )
            )

        return reranked_candidates

    def _score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        return list(
            self._model.rerank(
                query=query,
                documents=documents,
            )
        )