
from app.agents.state import AgentState
import logfire
from langchain_groq import ChatGroq
from app.config import settings
import re


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
    max_tokens=1024,
)


def get_message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def get_message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "user")).lower()
    return str(getattr(message, "type", "user")).lower()


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])

    history_lines = []
    for message in messages[:-1]:
        role = get_message_role(message)
        content = get_message_content(message)
        speaker = (
            "Assistant"
            if role in ("assistant", "ai")
            else "User"
        )
        history_lines.append(f"{speaker}: {content}")

    history = "\n".join(history_lines)

    latest_message = messages[-1] if messages else None
    user_message = (
        get_message_content(latest_message)
        if latest_message is not None
        else state.get("current_query", "")
    )

    prompt = f"""
You are the intent classifier for an enterprise AI agent.

Use BOTH the conversation history and the latest user message.
Determine what the user is asking now, including references to earlier turns.

Conversation history:
{history if history else "(No previous conversation messages.)"}

Latest user message:
{user_message}

Choose exactly one intent:

CONVERSATIONAL:
Greetings, ordinary conversation, requests to rephrase or explain
a previous answer, follow-up questions answerable from conversation
history, and questions asking what the user previously stated.
Use this route when the answer is available from the conversation
history and no external tool or knowledge-base lookup is needed.

CALCULATOR:
The user requests a mathematical calculation.

WEB_SEARCH:
The user needs current information or explicitly requests an internet search.

RETRIEVAL:
The user needs information from the local knowledge base, documents,
or technical evidence not established by the conversation history.
Use this route for document questions and factual questions requiring
retrieved evidence.

Important routing rules:
- Resolve pronouns and references such as "it", "that", and "the first
  approach" using conversation history.
- Do not route a question to RETRIEVAL merely because it is phrased
  as a factual question if the answer is already stated in conversation.
- Do not route a request to CONVERSATIONAL if it genuinely requires
  calculation, current web information, or local document retrieval.
- Never assume that missing conversation details exist.
- Return only one label.

Valid labels:
CONVERSATIONAL
CALCULATOR
WEB_SEARCH
RETRIEVAL
"""

    with logfire.span("Classifying user intent"):
        decision = str(llm.invoke(prompt).content).strip().upper()

        match = re.search(
            r"\b(CONVERSATIONAL|CALCULATOR|WEB_SEARCH|MCP_TOOL|RETRIEVAL)\b",
            decision,
        )
        decision = match.group(1) if match else "RETRIEVAL"

        logfire.info("Intent identified", route=decision)

    if decision == "CALCULATOR":
        expression = user_message.strip().lower()
        expression = re.sub(
            r"^(what is|calculate|compute|evaluate|please calculate)\s+",
            "",
            expression,
        ).rstrip(" ?.")

        return {
            "route": decision,
            "current_query": expression,
            "status": "Calculating expression",
            "plan": ["Intent: Calculator", f"Expression: {expression}"],
        }

    if decision == "MCP_TOOL":\n        return {\n            "route": decision,\n            "current_query": user_message,\n            "status": "Selecting approved MCP tool",\n            "plan": ["Intent: MCP Tool", f"Request: {user_message}"],\n        }\n\n    if decision == "WEB_SEARCH":
        return {
            "route": decision,
            "current_query": user_message,
            "status": "Searching the web",
            "plan": [
                "Intent: Web Search",
                f"Search query: {user_message}",
            ],
        }

    if decision == "CONVERSATIONAL":
        return {
            "route": decision,
            "current_query": user_message,
            "status": "Handling conversational request",
            "plan": [
                "Intent: Conversational",
                "Conversation context considered",
            ],
        }

    return {
        "route": "RETRIEVAL",
        "current_query": user_message,
        "status": "Technical query detected",
        "plan": [
            "Intent: Retrieval",
            f"Search: {user_message}",
        ],
    }
