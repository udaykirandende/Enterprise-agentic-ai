
import logfire

from app.agents.state import AgentState
from app.services.retrival.qdrant_service import search_enterprise_knowledge
from app.services.retrival.ranking_service import rerank_documents_with_scores


CANDIDATE_TOP_K = 25
RERANK_TOP_K = 5


def retrieve_node(state: AgentState) -> dict:
    """Retrieve candidates from Qdrant and rerank them with FlashRank."""
    query = state.get("current_query", "").strip()
    plan = state.get("plan", [])

    if not query:
        return {
            "documents": [],
            "status": "Empty retrieval query",
            "plan": plan + ["Retrieval skipped: empty query"],
        }

    try:
        with logfire.span("Knowledge retrieval"):
            raw_results = search_enterprise_knowledge(
                query,
                limit=CANDIDATE_TOP_K,
            )

            # Preserve each chunk's source and vector score.
            candidates = []
            for item in raw_results:
                content = item.get("content", "").strip()
                if content:
                    candidates.append({
                        "content": content,
                        "source": item.get("source", "Unknown"),
                        "vector_score": item.get("score"),
                    })

            logfire.info(
                "Qdrant retrieval completed",
                candidate_count=len(candidates),
                requested_top_k=CANDIDATE_TOP_K,
            )

        ranked = rerank_documents_with_scores(
            query=query,
            documents=[item["content"] for item in candidates],
            top_n=RERANK_TOP_K,
        )

        # Map the ranked text back to source metadata.
        # A list handles repeated identical chunk text safely.
        remaining = candidates.copy()
        formatted_docs = []

        for item in ranked:
            match_index = next(
                (
                    index for index, candidate in enumerate(remaining)
                    if candidate["content"] == item["text"]
                ),
                None,
            )

            if match_index is None:
                source = "Unknown"
                vector_score = None
            else:
                matched = remaining.pop(match_index)
                source = matched["source"]
                vector_score = matched["vector_score"]

            formatted_docs.append(
                f"SOURCE: {source}\n"
                f"VECTOR_SCORE: {vector_score}\n"
                f"RERANK_SCORE: {item['rerank_score']}\n"
                f"CONTENT: {item['text']}"
            )

        logfire.info(
            "Knowledge retrieval and reranking completed",
            retrieved_count=len(candidates),
            final_count=len(formatted_docs),
        )

        return {
            "documents": formatted_docs,
            "status": (
                f"Retrieved {len(candidates)} candidates; "
                f"selected {len(formatted_docs)} after reranking."
            ),
            "plan": plan + ["Qdrant retrieval", "FlashRank reranking"],
        }

    except Exception as exc:
        logfire.error(
            "Knowledge retrieval failed",
            error_type=type(exc).__name__,
        )
        return {
            "documents": [],
            "status": "Retrieval failed",
            "plan": plan + ["Knowledge retrieval failed"],
        }
