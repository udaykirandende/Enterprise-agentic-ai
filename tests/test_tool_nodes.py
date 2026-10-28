from unittest.mock import patch

from app.agents.nodes.calculator import calculator_node
from app.agents.nodes.web_search import (
    extract_requested_domains,
    web_search_node,
)


def test_calculator_returns_result():
    state = {"current_query": "2 + 3", "plan": ["Start"]}

    with patch("app.agents.nodes.calculator.calculate", return_value=5):
        result = calculator_node(state)

    assert result["final_answer"] == "Result: 5"
    assert result["status"] == "Completed"
    assert "Calculator executed" in result["plan"]


def test_calculator_handles_invalid_expression():
    state = {"current_query": "invalid", "plan": []}

    with patch(
        "app.agents.nodes.calculator.calculate",
        side_effect=ValueError("Invalid expression"),
    ):
        result = calculator_node(state)

    assert result["status"] == "Tool input error"
    assert result["final_answer"].startswith("Unable to calculate expression:")


def test_domain_restriction_is_not_added_without_explicit_request():
    assert extract_requested_domains("Search for Python tutorials on example.com") == []


def test_explicit_domain_restriction_is_extracted():
    result = extract_requested_domains("Search only on example.com")

    assert result == ["example.com"]


def test_web_search_handles_empty_query():
    result = web_search_node({"current_query": "", "plan": ["Start"]})

    assert result["status"] == "Tool input error"
    assert result["documents"] == []


def test_web_search_formats_mocked_results():
    mock_response = {
        "results": [
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org/",
                "content": "Official Python documentation.",
            }
        ]
    }

    with patch(
        "app.agents.nodes.web_search.TavilySearch"
    ) as mock_search:
        mock_search.return_value.invoke.return_value = mock_response
        result = web_search_node(
            {"current_query": "Python documentation", "plan": ["Start"]}
        )

    assert result["status"] == "Completed"
    assert len(result["documents"]) == 1
    assert "https://docs.python.org/" in result["documents"][0]
    assert "Web search executed" in result["plan"]
from unittest.mock import patch

from app.agents.nodes.retriver import retrieve_node


@patch("app.agents.nodes.retriver.rerank_documents_with_scores")
@patch("app.agents.nodes.retriver.search_enterprise_knowledge")
def test_retrieval_preserves_source_and_scores(mock_search, mock_rerank):
    mock_search.return_value = [
        {
            "content": "Jon Snow is a member of the Night's Watch.",
            "source": "game_of_thrones.pdf",
            "score": 0.82,
        },
        {
            "content": "Samwell Tarly becomes Jon's friend.",
            "source": "game_of_thrones.pdf",
            "score": 0.76,
        },
    ]
    mock_rerank.return_value = [
        {
            "text": "Samwell Tarly becomes Jon's friend.",
            "rerank_score": 0.99,
        },
        {
            "text": "Jon Snow is a member of the Night's Watch.",
            "rerank_score": 0.91,
        },
    ]

    result = retrieve_node(
        {"current_query": "Who is Jon Snow's friend?", "plan": ["Start"]}
    )

    assert len(result["documents"]) == 2
    assert result["documents"][0].startswith("SOURCE: game_of_thrones.pdf")
    assert "VECTOR_SCORE: 0.76" in result["documents"][0]
    assert "RERANK_SCORE: 0.99" in result["documents"][0]
    assert result["documents"][0].endswith(
        "Samwell Tarly becomes Jon's friend."
    )
    assert result["status"] == "Retrieved 2 relevant chunks."
    mock_search.assert_called_once_with(
        "Who is Jon Snow's friend?", limit=25
    )


@patch("app.agents.nodes.retriver.rerank_documents_with_scores")
@patch("app.agents.nodes.retriver.search_enterprise_knowledge")
def test_retrieval_skips_empty_query(mock_search, mock_rerank):
    result = retrieve_node(
        {"current_query": "   ", "plan": ["Start"]}
    )

    assert result["documents"] == []
    assert result["status"] == "Empty retrieval query"
    assert result["plan"] == [
        "Start",
        "Retrieval skipped: empty query",
    ]
    mock_search.assert_not_called()
    mock_rerank.assert_not_called()


@patch("app.agents.nodes.retriver.rerank_documents_with_scores")
@patch("app.agents.nodes.retriver.search_enterprise_knowledge")
def test_retrieval_handles_search_failure(mock_search, mock_rerank):
    mock_search.side_effect = RuntimeError("Qdrant unavailable")

    result = retrieve_node(
        {"current_query": "Who is Jon Snow?", "plan": ["Start"]}
    )

    assert result["documents"] == []
    assert result["status"] == "Retrieval failed"
    assert result["plan"] == [
        "Start",
        "Knowledge retrieval failed",
    ]
    mock_rerank.assert_not_called()


@patch("app.agents.nodes.retriver.rerank_documents_with_scores")
@patch("app.agents.nodes.retriver.search_enterprise_knowledge")
def test_retrieval_ignores_blank_chunks(mock_search, mock_rerank):
    mock_search.return_value = [
        {"content": "   ", "source": "empty.pdf", "score": 0.9},
        {"content": "Useful passage.", "source": "notes.pdf", "score": 0.8},
    ]
    mock_rerank.return_value = [
        {"text": "Useful passage.", "rerank_score": 0.95}
    ]

    result = retrieve_node(
        {"current_query": "Find useful information", "plan": []}
    )

    mock_rerank.assert_called_once_with(
        query="Find useful information",
        documents=["Useful passage."],
        top_n=5,
    )
    assert len(result["documents"]) == 1
    assert result["documents"][0].startswith("SOURCE: notes.pdf")
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents.nodes.responder import generate_node


@patch("app.agents.nodes.responder.get_langchain_llm")
def test_rag_responder_uses_retrieved_context(mock_get_llm):
    mock_get_llm.return_value.invoke.return_value = SimpleNamespace(
        content="Jon is a member of the Night's Watch."
    )

    result = generate_node({
        "route": "RETRIEVAL",
        "current_query": "Who is Jon Snow?",
        "documents": [
            "SOURCE: got.pdf\nCONTENT: Jon Snow joins the Night's Watch."
        ],
        "messages": [{"role": "user", "content": "Who is Jon Snow?"}],
        "plan": ["Start"],
    })

    prompt = mock_get_llm.return_value.invoke.call_args.args[0]
    assert "Jon Snow joins the Night's Watch" in prompt
    assert "Answer ONLY from the provided context" in prompt
    assert result["final_answer"] == "Jon is a member of the Night's Watch."
    assert result["status"] == "Completed"
    assert result["messages"][0]["role"] == "assistant"


@patch("app.agents.nodes.responder.get_langchain_llm")
def test_conversational_responder_includes_history(mock_get_llm):
    mock_get_llm.return_value.invoke.return_value = SimpleNamespace(
        content="Your project uses RAG."
    )

    result = generate_node({
        "route": "CONVERSATIONAL",
        "current_query": "What does my project use?",
        "documents": [],
        "messages": [
            {"role": "user", "content": "I am building a RAG project."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What does my project use?"},
        ],
        "plan": ["Start"],
    })

    prompt = mock_get_llm.return_value.invoke.call_args.args[0]
    assert "I am building a RAG project." in prompt
    assert "Understood." in prompt
    assert "Answer naturally." in prompt
    assert result["final_answer"] == "Your project uses RAG."


@patch("app.agents.nodes.responder.get_langchain_llm")
def test_responder_propagates_llm_failure(mock_get_llm):
    mock_get_llm.return_value.invoke.side_effect = RuntimeError("Gateway unavailable")

    with pytest.raises(RuntimeError, match="Gateway unavailable"):
        generate_node({
            "route": "RETRIEVAL",
            "current_query": "Who is Jon Snow?",
            "documents": [],
            "messages": [{"role": "user", "content": "Who is Jon Snow?"}],
            "plan": [],
        })
