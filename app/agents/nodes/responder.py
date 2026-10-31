
import logfire
from typing import Any

from app.agents.state import AgentState
from app.gateway import get_langchain_llm


def _get_message_content(message: Any) -> str:
    """Extract content from dictionary or LangChain message objects."""
    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def _get_message_role(message: Any) -> str:
    """Extract the role from dictionary or LangChain message objects."""
    if isinstance(message, dict):
        return str(message.get("role", "user"))

    return str(getattr(message, "type", "user"))


def _build_history(messages: list[Any]) -> str:
    """Format conversation history for the LLM."""
    history_lines = []

    for message in messages[:-1]:
        role = _get_message_role(message)
        content = _get_message_content(message)

        speaker = "Assistant" if role in ("assistant", "ai") else "User"
        history_lines.append(f"{speaker}: {content}")

    return "\n".join(history_lines)


def _build_web_evidence(documents: list[Any]) -> str:
    """Format web-search results without discarding source information."""
    if not documents:
        return "No web-search evidence was provided."

    return "\n\n".join(
        f"[SOURCE {index}]\n{document}"
        for index, document in enumerate(documents, start=1)
    )


def generate_node(state: AgentState) -> dict[str, Any]:
    """Generate conversational, RAG-grounded, or web-grounded responses."""

    route = state.get("route", "RETRIEVAL")
    query = state.get("current_query", "")
    messages = state.get("messages", [])
    history_str = _build_history(messages)

    user_msg = (
        _get_message_content(messages[-1])
        if messages
        else query
    )

    # Stop before another LLM call when retrieval has no verified evidence.
    retrieval_stop_statuses = {
        "No relevant evidence found",
        "Relevance check failed",
        "Retrieval failed",
        "Empty retrieval query",
    }

    status = state.get("status", "")

    if route == "RETRIEVAL" and status in retrieval_stop_statuses:
        answer = state.get("final_answer") or (
            "I couldn't find verified, relevant information in the knowledge base."
        )
        plan = state.get("plan", [])

        logfire.info(
            "RAG response returned without LLM call",
            status=status,
        )

        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "status": status,
            "plan": plan + [
                "Stopped before response generation: no verified evidence"
            ],
            "documents": [],
        }

    if route == "CONVERSATIONAL":
        logfire.info("Generating conversational response.")

        prompt = f"""
You are a helpful enterprise AI assistant.

Instructions:
- Answer the user's exact question.
- Use conversation history when relevant.
- Do not invent personal information or previous statements.
- Be concise and natural.
- Do not add unnecessary sections, tables, or learning plans.
- Ensure the answer is complete.

Conversation history:
{history_str}

Latest user message:
{user_msg}
"""

    elif route == "WEB_SEARCH":
        logfire.info("Synthesizing web-search evidence.")

        evidence = _build_web_evidence(state.get("documents", []))

        prompt = f"""
You are an evidence-grounded enterprise research assistant.

Your task is to answer the user's question using the supplied
web-search evidence.

IMPORTANT EVIDENCE RULES:

1. SOURCE RELIABILITY
- Prefer primary and authoritative sources.
- Examples include official government websites, official
  documentation, company websites, regulatory filings, and
  original research.
- Treat Wikipedia, blogs, aggregators, and news summaries as
  secondary sources.
- Do not treat a secondary source as official confirmation.
- Do not claim that a source is authoritative unless its
  identity and content support that conclusion.

2. FACTUAL SUPPORT
- Every factual claim must be supported by the supplied evidence.
- Do not infer facts that the source does not establish.
- A search-result snippet is limited evidence. Do not claim
  to have verified information that is absent from the snippet.
- If evidence is insufficient, explicitly state what is missing.
- Never fabricate dates, statistics, quotations, or citations.

3. DATES AND CURRENT INFORMATION
- Distinguish publication dates, event dates, release dates,
  update dates, and projected dates.
- Never infer an exact release date from a month or year.
- For software releases, distinguish stable releases from
  alpha, beta, development, and release-candidate versions.
- Do not call a version stable unless the evidence supports it.
- If sources disagree about a date or current status, explain
  the disagreement instead of selecting an unsupported answer.

4. DEFINITIONS AND MEASUREMENTS
- For population statistics, distinguish administrative city
  populations, urban populations, and metropolitan populations.
- For financial statistics, distinguish estimates, actual
  results, forecasts, and projections.
- Do not compare figures that use different definitions as
  though they measure the same thing.
- State the relevant year and measurement boundary.

5. SOURCE CONFLICTS
- Identify contradictions between sources.
- Prefer the most relevant primary source when available.
- If the conflict cannot be resolved from the supplied evidence,
  disclose that limitation.
- Do not silently combine incompatible figures.

6. CITATIONS
- Use only URLs included in the supplied evidence.
- Cite claims with numbered references such as [1] and [2].
- Number references according to the SOURCE entries below.
- Do not invent URLs or cite sources that were not supplied.
- Include a Sources section containing the source title and
  exact URL.
- Attach citations to the claims they support.

7. RESPONSE STYLE
- Answer the question directly.
- Give the most useful supported answer first.
- Include qualifications only when they materially affect
  the correctness of the answer.
- Do not expose internal reasoning or describe hidden
  thought processes.
- If no reliable answer can be established, say so clearly.

User question:
{user_msg}

Web-search evidence:
{evidence}
"""

    else:
        logfire.info("Generating technical RAG response.")

        documents = state.get("documents", [])
        context = "\n\n".join(documents)

        prompt = f"""
You are a senior technical AI assistant.

Answer ONLY using the supplied knowledge-base context.

Instructions:
- Ground factual claims in the provided documents.
- Do not invent facts, sources, or quotations.
- Do not use external knowledge to fill gaps.
- If the context does not support an answer, clearly state
  that the information is unavailable in the documents.
- Preserve important technical details and terminology.
- Cite document names when they are available.
- Give a direct and complete answer.

Technical context:
{context}

Conversation history:
{history_str}

User question:
{user_msg}
"""

    with logfire.span("LLM Response Generation"):
        try:
            response = get_langchain_llm(feature="rag").invoke(prompt)
            answer = response.content

            if not isinstance(answer, str):
                answer = str(answer)

            plan = state.get("plan", [])

            logfire.info(
                "LLM response generated",
                route=route,
            )

            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "status": "Completed",
                "plan": plan + (
                    ["Web results synthesized with evidence rules"]
                    if route == "WEB_SEARCH"
                    else ["Response generated"]
                ),
            }

        except Exception as exc:
            logfire.error(
                "LLM response generation failed",
                route=route,
                error_type=type(exc).__name__,
            )
            raise
