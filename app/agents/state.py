
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    route: str
    documents: list[str]
    final_answer: str
    plan: list[str]
    status: str
    tool_result: str
