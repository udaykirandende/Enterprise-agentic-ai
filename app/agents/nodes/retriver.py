import logfire

from app.agents.state import AgentState
from app.services.retrival.qdrant_service import search_enterprise_knowledge
from app.services.retrival.ranking_service import rerank_documents_with_scores


CANDIDATE_TOP_K = 25
RERANK_TOP_K = 5

# Initial cutoff based on the observed relevant (~0.64) and irrelevant
# (~0.00001) examples. Validate against more queries before treating it as final.
MIN_RERANK_SCORE = 0.1


def _no_evidence_result(plan: list[str]) -> dict:
    return {
        "documents": [],
        "final_answer": (
            "I couldn't find relevant information in the knowledge-base "
            "documents to answer that question."
        ),
        "status": "No relevant evidence found",
        "plan": plan + ["Relevance check: no chunks passed the score threshold"],
    }


def retrieve_node(state: AgentState) -> dict:
    """Retrieve, rerank, and reject chunks that do not meet the relevance cutoff."""
    query = state.get("current_query", "").strip()
    plan = state.get("plan", [])

    if not query:
        return {
            "documents": [],
            "final_answer": "Please provide a question to search the knowledge base.",
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

        if not candidates:
            return _no_evidence_result(plan + ["Qdrant retrieval"])

        ranked = rerank_documents_with_scores(
            query=query,
            documents=[item["content"] for item in candidates],
            top_n=RERANK_TOP_K,
        )

        # The reranker returns None scores if it fails. Do not pass unverified
        # candidates to the LLM when relevance could not be assessed.
        scored_items = [
            item for item in ranked
            if isinstance(item.get("rerank_score"), (int, float))
        ]
        if not scored_items:
            logfire.warning(
                "Relevance check could not be completed",
                candidate_count=len(candidates),
            )
            return {
                "documents": [],
                "final_answer": (
                    "I couldn't verify document relevance because the reranking "
                    "step failed. Please try again."
                ),
                "status": "Relevance check failed",
                "plan": plan + [
                    "Qdrant retrieval",
                    "FlashRank reranking failed or returned no scores",
                ],
            }

        accepted = [
            item for item in scored_items
            if item["rerank_score"] >= MIN_RERANK_SCORE
        ]

        logfire.info(
            "RAG relevance threshold applied",
            candidate_count=len(candidates),
            reranked_count=len(scored_items),
            accepted_count=len(accepted),
            minimum_rerank_score=MIN_RERANK_SCORE,
        )

        if not accepted:
            return _no_evidence_result(
                plan + [
                    "Qdrant retrieval",
                    "FlashRank reranking",
                    f"Relevance threshold: {MIN_RERANK_SCORE}",
                ]
            )

        # Map accepted ranked text back to source metadata. A list handles
        # repeated identical chunk text safely.
        remaining = candidates.copy()
        formatted_docs = []
        accepted_texts = {item["text"] for item in accepted}

        for item in ranked:
            if item.get("text") not in accepted_texts:
                continue
            if not isinstance(item.get("rerank_score"), (int, float)):
                continue

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
            "Relevant knowledge chunks selected",
            retrieved_count=len(candidates),
            accepted_count=len(formatted_docs),
            minimum_rerank_score=MIN_RERANK_SCORE,
        )

        return {
            "documents": formatted_docs,
            "status": f"Retrieved {len(formatted_docs)} relevant chunks.",
            "plan": plan + [
                "Qdrant retrieval",
                "FlashRank reranking",
                f"Relevance threshold passed: {MIN_RERANK_SCORE}",
            ],
        }

    except Exception as exc:
        logfire.error(
            "Knowledge retrieval failed",
            error_type=type(exc).__name__,
        )
        return {
            "documents": [],
            "final_answer": (
                "Knowledge-base retrieval failed, so I couldn't verify an answer. "
                "Please try again."
            ),
            "status": "Retrieval failed",
            "plan": plan + ["Knowledge retrieval failed"],
        }
