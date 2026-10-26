"""Optional integration test for durable LangGraph checkpoints.

Set DATABASE_URL to a disposable PostgreSQL database to run this test. The
test writes only to a uniquely named thread ID.
"""
import os
import uuid

import pytest

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="Set DATABASE_URL to run PostgreSQL checkpoint integration test",
)


def test_checkpoint_survives_checkpointer_recreation():
    from langgraph.checkpoint.postgres import PostgresSaver
    from app.agents.graph import build_graph

    thread_id = f"pytest-persistence-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
        graph.update_state(
            config,
            {
                "current_query": "checkpoint persistence probe",
                "plan": ["checkpoint-survives-reopen"],
                "status": "test",
            },
        )
        state = graph.get_state(config)
        assert state.values["plan"] == ["checkpoint-survives-reopen"]

    with PostgresSaver.from_conn_string(DATABASE_URL) as reopened_checkpointer:
        reopened_checkpointer.setup()
        reopened_graph = build_graph(reopened_checkpointer)
        restored = reopened_graph.get_state(config)
        assert restored.values["plan"] == ["checkpoint-survives-reopen"]
