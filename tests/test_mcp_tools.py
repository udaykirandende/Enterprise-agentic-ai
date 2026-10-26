from unittest.mock import patch

from app.agents.nodes.mcp_tools import execute_mcp_request, mcp_tool_node


def test_mcp_returns_clear_message_when_not_configured(monkeypatch):
    monkeypatch.delenv("MCP_SERVERS_JSON", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_TOOLS", raising=False)

    result = execute_mcp_request("Find an item")

    assert result["status"] == "MCP not configured"
    assert "MCP_SERVERS_JSON" in result["final_answer"]


def test_mcp_requires_allowlist(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '{"example":{"transport":"streamable_http","url":"https://example.com/mcp"}}',
    )
    monkeypatch.delenv("MCP_ALLOWED_TOOLS", raising=False)

    result = execute_mcp_request("Find an item")

    assert result["status"] == "MCP blocked"
    assert "allowlisted" in result["final_answer"]


def test_mcp_node_skips_empty_query():
    result = mcp_tool_node({"current_query": "  ", "plan": ["Start"]})

    assert result["status"] == "Tool input error"
    assert result["plan"][-1] == "MCP skipped: empty request"


def test_mcp_node_appends_execution_trace():
    with patch(
        "app.agents.nodes.mcp_tools.execute_mcp_request",
        return_value={
            "status": "Completed",
            "final_answer": "Tool result",
            "plan": ["MCP tool executed: approved_tool"],
        },
    ):
        result = mcp_tool_node(
            {"current_query": "Use the approved tool", "plan": ["Intent: MCP Tool"]}
        )

    assert result["final_answer"] == "Tool result"
    assert result["plan"][-1] == "MCP tool executed: approved_tool"
