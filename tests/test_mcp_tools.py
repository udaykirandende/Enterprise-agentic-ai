from types import SimpleNamespace
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


def test_mcp_executes_only_allowlisted_tool(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '{"example":{"transport":"streamable_http","url":"https://example.com/mcp"}}',
    )
    monkeypatch.setenv("MCP_ALLOWED_TOOLS", "approved_tool")

    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, args):
            calls.append((self.name, args))
            return {"result": "ok"}

    approved = FakeTool("approved_tool")
    disallowed = FakeTool("disallowed_tool")

    class FakeClient:
        def __init__(self, servers):
            assert "example" in servers

        async def get_tools(self):
            return [approved, disallowed]

    class FakeModel:
        def bind_tools(self, tools):
            self.bound_tool_names = [tool.name for tool in tools]
            return self

        async def ainvoke(self, messages):
            return SimpleNamespace(
                tool_calls=[
                    {"name": "approved_tool", "args": {"query": "sample"}}
                ],
                content="",
            )

    model = FakeModel()
    with (
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            FakeClient,
        ),
        patch("app.gateway.get_langchain_llm", return_value=model),
    ):
        result = execute_mcp_request("Use the approved tool")

    assert model.bound_tool_names == ["approved_tool"]
    assert calls == [("approved_tool", {"query": "sample"})]
    assert result["status"] == "Completed"
    assert result["plan"] == ["MCP tool executed: approved_tool"]
