
import re
from urllib.parse import urlparse

import logfire
from langchain_tavily import TavilySearch

from app.agents.state import AgentState


def extract_requested_domains(query: str) -> list[str]:
    """Restrict search only when the user explicitly requests specific domains."""
    if not re.search(
        r"\b(only|restrict(?:ed)? to|limit(?:ed)? to|exclusively)\b",
        query,
        re.IGNORECASE,
    ):
        return []

    candidates = re.findall(
        r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
        query,
    )

    domains = []
    for candidate in candidates:
        domain = urlparse(f"https://{candidate}").hostname
        if domain and domain not in domains:
            domains.append(domain)

    return domains


def _clean_text(value: object, limit: int = 1200) -> str:
    """Normalize whitespace and bound untrusted search-result text."""
    if not isinstance(value, str):
        return ""

    text = re.sub(r"\s+", " ", value).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def _valid_http_url(value: object) -> str:
    """Accept only absolute HTTP(S) URLs with a hostname."""
    if not isinstance(value, str):
        return ""

    url = value.strip()
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
    except ValueError:
        return ""

    return url


def web_search_node(state: AgentState) -> dict:
    query = state.get("current_query", "").strip()
    plan = state.get("plan", [])

    if not query:
        message = "Web search requires a non-empty query."
        return {
            "tool_result": message,
            "final_answer": message,
            "documents": [],
            "status": "Tool input error",
            "plan": plan,
        }

    try:
        requested_domains = extract_requested_domains(query)

        search_options = {
            "max_results": 5,
            "topic": "general",
            "search_depth": "advanced",
            "include_answer": False,
        }

        if requested_domains:
            search_options["include_domains"] = requested_domains

        search_tool = TavilySearch(**search_options)
        response = search_tool.invoke({"query": query})

        if isinstance(response, list):
            results = response
        elif isinstance(response, dict):
            results = response.get("results", [])
        else:
            raise ValueError("Unexpected Tavily response format.")

        documents = []
        seen_urls = set()

        for item in results:
            if not isinstance(item, dict):
                continue

            url = _valid_http_url(item.get("url"))
            if not url:
                continue

            canonical_url = url.rstrip("/").lower()
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            title = _clean_text(item.get("title")) or "Untitled source"
            snippet = _clean_text(
                item.get("content")
                or item.get("snippet")
                or item.get("raw_content")
                or ""
            )

            documents.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

        if not documents:
            message = "No usable web search results were returned."
            status = "No results"
        else:
            scope = (
                f"Restricted domains: {', '.join(requested_domains)}"
                if requested_domains
                else "Web-wide search"
            )

            # Keep source evidence structured. A separate responder can
            # synthesize an answer from this evidence without losing URLs.
            sections = [
                f"Search query: {query}",
                f"Search scope: {scope}",
                "Search snippets are evidence to assess, not independently verified facts.",
            ]

            for index, document in enumerate(documents, start=1):
                sections.append(
                    f"{index}. {document['title']}\n"
                    f"URL: {document['url']}\n"
                    f"Snippet: {document['snippet'] or 'No snippet provided.'}"
                )

            message = "\n\n".join(sections)
            status = "Completed"

        logfire.info(
            "Web search completed",
            result_count=len(documents),
            restricted_domains=requested_domains,
        )

        formatted_documents = [
            (
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Snippet: {item['snippet']}"
            )
            for item in documents
        ]

        return {
            "tool_result": message,
            "final_answer": message,
            "documents": formatted_documents,
            "status": status,
            "plan": plan + ["Web search executed"],
        }

    except Exception as exc:
        logfire.exception(
    "Web search failed",
    error_type=type(exc).__name__,
    error_message=str(exc),
   )

        message = (
            "Web search failed. Check the Tavily API key, "
            "network connection, and Tavily service."
        )

        return {
            "tool_result": message,
            "final_answer": message,
            "documents": [],
            "status": "Tool error",
            "plan": plan + ["Web search failed"],
        }
