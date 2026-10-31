
import os
import uuid

import pytest
import requests


BASE_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/query")


@pytest.fixture
def thread_ids():
    """Generate isolated thread IDs for each test."""
    suffix = uuid.uuid4().hex
    return {
        "memory": f"pytest-memory-{suffix}",
        "isolated": f"pytest-isolated-{suffix}",
    }


def ask(question: str, thread_id: str) -> dict:
    """Send a question to the running agent API."""
    response = requests.post(
        BASE_URL,
        json={"q": question, "thread_id": thread_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def test_memory_recall_same_thread(thread_ids):
    """A fact given in one turn should be recalled in the next."""
    fact = f"Memory test codeword is Recall{uuid.uuid4().hex[:10]}"

    first = ask(
        f"Remember this fact for our conversation: {fact}.",
        thread_ids["memory"],
    )
    assert first.get("status") == "Completed"

    second = ask(
        "What is the memory test codeword I told you to remember?",
        thread_ids["memory"],
    )

    assert second.get("status") == "Completed"
    assert fact.split()[-1].lower().rstrip(".") in (
        second.get("answer", "").lower()
    ), f"Expected the agent to recall the codeword. Answer: {second.get('answer')}"


def test_memory_isolated_between_threads(thread_ids):
    """A different thread must not inherit another thread's fact."""
    codeword = f"Private{uuid.uuid4().hex[:12]}"

    first = ask(
        f"Remember this fact: the secret test codeword is {codeword}.",
        thread_ids["memory"],
    )
    assert first.get("status") == "Completed"

    second = ask(
        "What secret test codeword did I tell you earlier?",
        thread_ids["isolated"],
    )

    answer = second.get("answer", "").lower()
    assert codeword.lower() not in answer, (
        "A separate thread unexpectedly revealed another thread's codeword."
    )


def test_natural_followup_uses_conversation_history(thread_ids):
    topic = f"Project Nebula{uuid.uuid4().hex[:8]}"

    first = ask(
        f"My project is called {topic}. It is a document Q&A assistant.",
        thread_ids["memory"],
    )
    assert first.get("status") == "Completed"

    second = ask(
        "Suggest a short tagline for it.",
        thread_ids["memory"],
    )
    assert second.get("status") == "Completed"

    answer = second.get("answer", "").lower()

    assert any(
        term in answer
        for term in ("document", "docs", "q&a", "question")
    ), (
        "The response did not appear to use the earlier project context. "
        f"Answer: {second.get('answer')}"
    )
