import os
import uuid
from contextlib import AbstractContextManager
from typing import Optional

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

from app.agents import graph as graph_module
from app.guardrails import initialize_rails, guard

app = FastAPI(title="Enterprise Agentic RAG API")
_checkpointer_context: Optional[AbstractContextManager] = None


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=10000)
    # Omitted IDs get a fresh isolated conversation; clients must reuse the
    # returned thread_id to continue a conversation.
    thread_id: Optional[str] = Field(default=None, min_length=1, max_length=200)


@app.on_event("startup")
def startup_event():
    global _checkpointer_context
    initialize_rails()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        from langgraph.checkpoint.postgres import PostgresSaver

        context = PostgresSaver.from_conn_string(database_url)
        try:
            checkpointer = context.__enter__()
            checkpointer.setup()
            graph_module.configure_checkpointer(checkpointer)
            _checkpointer_context = context
            logfire.info("Persistent PostgreSQL checkpointer initialized")
        except Exception:
            context.__exit__(*__import__("sys").exc_info())
            raise
    else:
        logfire.warning(
            "DATABASE_URL is not configured; conversation memory is in-memory "
            "and will be lost when the process restarts"
        )


@app.on_event("shutdown")
def shutdown_event():
    global _checkpointer_context
    if _checkpointer_context is not None:
        _checkpointer_context.__exit__(None, None, None)
        _checkpointer_context = None


@app.get("/")
def home():
    return {
        "message": "Enterprise Agentic RAG API is live.",
        "memory_backend": "postgres" if os.getenv("DATABASE_URL") else "memory",
        "mcp_configured": bool(os.getenv("MCP_SERVERS_JSON")),
    }


@app.get("/graph")
def get_graph_image():
    """Return a PNG image of the workflow graph."""
    try:
        png_bytes = graph_module.rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:
        logfire.warning("Could not render graph image", error_type=type(exc).__name__)
        return {"error": "Could not generate graph image."}


@app.post("/query")
def query(request: QueryRequest):
    q = request.q.strip()
    if not q:
        return {
            "question": request.q,
            "answer": "Please provide a non-empty question.",
            "thought_process": [],
            "status": "Tool input error",
            "sources": [],
        }

    thread_id = request.thread_id or str(uuid.uuid4())
    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "route": "",
        "documents": [],
        "final_answer": "",
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "tool_result": "",
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info("Request blocked by guardrails", thread_id=thread_id)
            return {
                "question": q,
                "thread_id": thread_id,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval Skipped"],
                "status": "Blocked",
                "sources": [],
            }

        final_output = graph_module.rag_agent.invoke(initial_state, config=config)
        return {
            "question": q,
            "thread_id": thread_id,
            "answer": final_output.get("final_answer", ""),
            "thought_process": final_output.get("plan", []),
            "status": final_output.get("status", ""),
            "sources": final_output.get("documents", []),
        }
    except Exception as exc:
        logfire.exception("Backend execution failed", error_type=type(exc).__name__)
        return {
            "question": q,
            "thread_id": thread_id,
            "answer": "The request could not be completed. Check application logs.",
            "thought_process": ["Execution Failed"],
            "status": "error",
            "sources": [],
        }
