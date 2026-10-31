
import re

import logfire
from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.config import settings


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
    max_tokens=1024,
)


VALID_INTENTS = {
    "CONVERSATIONAL",
    "CALCULATOR",
    "WEB_SEARCH",
    "MCP_TOOL",
    "RETRIEVAL",
}


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
    ).strip()

    prompt = f"""
You are a precise intent classifier for an enterprise AI agent.

Your task is to classify the LATEST user request.

The system follows a LOCAL-FIRST knowledge strategy:

1. Try the application's local knowledge base for general
   factual and knowledge-based questions.
2. If relevant local evidence is not found, the graph will
   automatically fall back to web search.
3. Explicit requests for current information or internet
   searches can go directly to web search.

Conversation history:
{history if history else "(No previous conversation messages.)"}

Latest user message:
{user_message}

Choose exactly ONE route.

1. CONVERSATIONAL

Choose this when:
- The user greets the assistant.
- The user engages in ordinary casual conversation.
- The user asks to rephrase or simplify a previous answer.
- The answer is already available in conversation history.
- The user asks a follow-up that depends on previous
  conversation context rather than external evidence.

Do NOT choose this merely because a question is short.

2. CALCULATOR

Choose this when:
- The user requests arithmetic.
- The user asks to calculate, evaluate, or compute an expression.
- The requested answer requires numerical computation.

3. WEB_SEARCH

Choose this when:
- The user explicitly requests an internet or web search.
- The user asks for current, latest, recent, or up-to-date information.
- The user requests live information.
- The user asks for current prices, weather, news, or live events.
- The user asks for the latest software version or release.
- The user explicitly requests external sources or online verification.
- The user asks to verify an unfamiliar entity's existence or
  an official release date.

Do NOT choose WEB_SEARCH just because a question is factual,
technical, about a public person, or about a company.

4. MCP_TOOL

Choose this only when:
- The user explicitly requests an action through a configured
  external MCP server.
- The request requires a capability provided by an available MCP tool.

Do not choose MCP_TOOL for ordinary web searches or calculations.

5. RETRIEVAL

Choose this for:
- General factual questions.
- Questions about people, places, organizations, and events.
- Questions about books, films, fictional characters, and stories.
- General technical and programming questions.
- Questions about concepts, definitions, and explanations.
- Questions that may be answered from uploaded documents.
- Questions about the application's local knowledge base.
- Questions where local evidence should be checked before
  using external web information.

Also choose RETRIEVAL when the user explicitly refers to an
uploaded PDF, file, or locally indexed document.

The graph will handle the next step:
- Relevant evidence found: answer using local RAG.
- No relevant evidence found: automatically use web search.
- Retrieval infrastructure failure: report the failure rather
  than silently treating it as a normal evidence miss.

ROUTING EXAMPLES:

"Who is Jon Snow?"
RETRIEVAL

"Who is Cristiano Ronaldo?"
RETRIEVAL

"Explain machine learning."
RETRIEVAL

"What is LangGraph?"
RETRIEVAL

"What is the latest stable LangGraph version?"
WEB_SEARCH

"Who won the 2024 Formula One Drivers' Championship?"
RETRIEVAL

"Who won today's Formula One race?"
WEB_SEARCH

"What is the current price of Bitcoin?"
WEB_SEARCH

"According to the uploaded PDF, who is Jon Snow?"
RETRIEVAL

"Find the answer in my Game of Thrones document."
RETRIEVAL

"Calculate 25 * (4 + 3)."
CALCULATOR

"Explain your previous answer in simpler words."
CONVERSATIONAL

"Hello, how are you?"
CONVERSATIONAL

"Search the internet for the latest LangGraph documentation."
WEB_SEARCH

"Use the configured MCP server to perform this action."
MCP_TOOL

FINAL RULES:
- Use conversation history to resolve pronouns and references.
- Do not classify general factual questions as conversational.
- Prefer RETRIEVAL for general knowledge questions.
- Prefer WEB_SEARCH for explicit internet searches and current information.
- Prefer CALCULATOR for numerical calculations.
- Prefer MCP_TOOL only for configured MCP actions.
- Return exactly one valid label and nothing else.

Valid labels:
CONVERSATIONAL
CALCULATOR
WEB_SEARCH
MCP_TOOL
RETRIEVAL
"""

    with logfire.span("Classifying user intent"):
        raw_decision = str(llm.invoke(prompt).content).strip().upper()

        match = re.search(
            r"\b(CONVERSATIONAL|CALCULATOR|WEB_SEARCH|MCP_TOOL|RETRIEVAL)\b",
            raw_decision,
        )

        if match:
            decision = match.group(1)
            classifier_fallback = False
        else:
            # Local-first default when the model returns an invalid label.
            decision = "RETRIEVAL"
            classifier_fallback = True

        logfire.info(
            "Intent identified",
            route=decision,
            classifier_fallback=classifier_fallback,
            raw_decision=raw_decision,
        )

    if decision == "CALCULATOR":
        expression = user_message.lower()

        expression = re.sub(
            r"^(what is|calculate|compute|evaluate|please calculate)\s+",
            "",
            expression,
        ).rstrip(" ?.")

        return {
            "route": decision,
            "current_query": expression,
            "status": "Calculating expression",
            "plan": [
                "Intent: Calculator",
                f"Expression: {expression}",
            ],
        }

    if decision == "MCP_TOOL":
        return {
            "route": decision,
            "current_query": user_message,
            "status": "Selecting approved MCP tool",
            "plan": [
                "Intent: MCP Tool",
                f"Request: {user_message}",
            ],
        }

    if decision == "WEB_SEARCH":
        plan = [
            "Intent: Web Search",
            f"Search query: {user_message}",
        ]

        if classifier_fallback:
            plan.append(
                "Invalid classifier output; defaulted to local retrieval."
            )

        return {
            "route": decision,
            "current_query": user_message,
            "status": "Searching the web",
            "plan": plan,
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
        "status": "Searching local knowledge base",
        "plan": [
            "Intent: Retrieval",
            f"Knowledge-base query: {user_message}",
        ],
    }
