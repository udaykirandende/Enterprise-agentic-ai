
import os
import time
import logfire
from flashrank import Ranker, RerankRequest

_ranker = None


def _get_ranker() -> Ranker:
    """Load the FlashRank model once and reuse it."""
    global _ranker

    if _ranker is None:
        cache_dir = os.getenv("FLASHRANK_CACHE_DIR", ".cache/flashrank")

        try:
            _ranker = Ranker(cache_dir=cache_dir)
        except Exception:
            logfire.warning(
                "FlashRank custom cache initialization failed; using default cache"
            )
            _ranker = Ranker()

    return _ranker


def rerank_documents_with_scores(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Return reranked documents with their FlashRank scores.

    Scores are for ranking candidates, not calibrated probabilities.
    """
    if not query.strip() or not documents or top_n <= 0:
        return []

    start_time = time.perf_counter()

    passages = [
        {"id": index, "text": document}
        for index, document in enumerate(documents)
        if isinstance(document, str) and document.strip()
    ]

    if not passages:
        return []

    try:
        ranker = _get_ranker()
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        ranked = [
            {
                "text": item["text"],
                "rerank_score": float(item["score"]),
            }
            for item in results[:top_n]
        ]

        logfire.info(
            "Reranking completed",
            candidate_count=len(passages),
            returned_count=len(ranked),
            duration_seconds=round(time.perf_counter() - start_time, 3),
            top_score=ranked[0]["rerank_score"] if ranked else None,
        )

        return ranked

    except Exception as exc:
        logfire.error(
            "Reranking failed",
            error_type=type(exc).__name__,
        )

        # Preserve availability when the reranker fails.
        return [
            {"text": item["text"], "rerank_score": None}
            for item in passages[:top_n]
        ]


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[str]:
    """Backward-compatible interface for existing callers."""
    return [
        item["text"]
        for item in rerank_documents_with_scores(query, documents, top_n)
    ]
