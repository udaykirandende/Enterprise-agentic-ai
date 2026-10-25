
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriver import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.calculator import calculator_node
from app.agents.nodes.web_search import web_search_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("web_search", web_search_node)


def route_planner(state: AgentState):
    route = state.get("route", "RETRIEVAL")

    if route == "CONVERSATIONAL":
        return "responder"
    if route == "CALCULATOR":
        return "calculator"
    if route == "WEB_SEARCH":
        return "web_search"
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
    },
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("web_search", "responder")
workflow.add_edge("responder", END)
workflow.add_edge("calculator", END)

checkpointer = MemorySaver()
rag_agent = workflow.compile(checkpointer=checkpointer)
