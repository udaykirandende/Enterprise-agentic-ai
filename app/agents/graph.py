
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriver import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.calculator import calculator_node
from app.agents.nodes.web_search import web_search_node
from app.agents.nodes.mcp_tools import mcp_tool_node


RETRIEVAL_MISS = "No relevant evidence found"

RETRIEVAL_FAILURES = {
    "Relevance check failed",
    "Retrieval failed",
    "Empty retrieval query",
}


def fallback_to_web_node(state: AgentState) -> dict:
    """
    Switch to web search only when local retrieval completes
    successfully but finds no relevant evidence.
    """
    plan = list(state.get("plan") or [])

    plan.append("Local knowledge base returned no relevant evidence.")
    plan.append("Falling back to web search.")

    return {
        "route": "WEB_SEARCH",
        "status": "Falling back to web search",
        "plan": plan,
    }


def route_planner(state: AgentState) -> str:
    """Choose the first node based on the planner's route."""
    route = state.get("route", "RETRIEVAL")

    routes = {
        "CONVERSATIONAL": "responder",
        "CALCULATOR": "calculator",
        "WEB_SEARCH": "web_search",
        "MCP_TOOL": "mcp_tool",
        "RETRIEVAL": "retriever",
    }

    return routes.get(route, "retriever")


def route_retrieval_result(state: AgentState) -> str:
    """
    Route retrieval results.

    Relevant evidence -> responder
    No relevant evidence -> web fallback
    Retrieval infrastructure failure -> responder
    """
    status = state.get("status", "")

    if status == RETRIEVAL_MISS:
        return "fallback_to_web"

    if status in RETRIEVAL_FAILURES:
        return "responder"

    return "responder"


def build_graph(checkpointer=None):
    """Build the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("mcp_tool", mcp_tool_node)
    workflow.add_node("fallback_to_web", fallback_to_web_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Planner routing
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

    # Retrieval result routing
    workflow.add_conditional_edges(
        "retriever",
        route_retrieval_result,
        {
            "fallback_to_web": "fallback_to_web",
            "responder": "responder",
        },
    )

    # Fallback and tool connections
    workflow.add_edge("fallback_to_web", "web_search")
    workflow.add_edge("web_search", "responder")

    # Terminal nodes
    workflow.add_edge("responder", END)
    workflow.add_edge("calculator", END)
    workflow.add_edge("mcp_tool", END)

    return workflow.compile(
        checkpointer=checkpointer or MemorySaver()
    )


# Default application graph
rag_agent = build_graph()


def configure_checkpointer(checkpointer):
    """Rebuild the active graph with a supplied checkpointer."""
    global rag_agent

    rag_agent = build_graph(checkpointer)

    return rag_agent
