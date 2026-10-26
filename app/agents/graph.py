from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriver import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.calculator import calculator_node
from app.agents.nodes.web_search import web_search_node
from app.agents.nodes.mcp_tools import mcp_tool_node


def build_graph(checkpointer=None):
    """Build the workflow with an injected checkpointer.

    MemorySaver is the safe local/test default. Production should pass a
    database-backed checkpointer (PostgresSaver) to persist state across restarts.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("mcp_tool", mcp_tool_node)

    def route_planner(state: AgentState):
        route = state.get("route", "RETRIEVAL")
        if route == "CONVERSATIONAL":
            return "responder"
        if route == "CALCULATOR":
            return "calculator"
        if route == "WEB_SEARCH":
            return "web_search"
        if route == "MCP_TOOL":
            return "mcp_tool"
        return "retriever"

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {
            "retriever": "retriever",
            "responder": "responder",
            "calculator": "calculator",
            "web_search": "web_search",
            "mcp_tool": "mcp_tool",
        },
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("web_search", "responder")
    workflow.add_edge("responder", END)
    workflow.add_edge("calculator", END)
    workflow.add_edge("mcp_tool", END)
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


rag_agent = build_graph()


def configure_checkpointer(checkpointer):
    """Replace the active graph with one using the supplied checkpointer."""
    global rag_agent
    rag_agent = build_graph(checkpointer)
    return rag_agent
