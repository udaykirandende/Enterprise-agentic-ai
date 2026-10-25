# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory.
    """

    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": q
            }
        ],
        "current_query": q,
        "route": "",
        "documents": [],
        "final_answer": "",
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # ---------- Guardrails ----------
        rail_fired, rail_response = guard(q)

        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")

            return {
                "question": q,
                "answer": rail_response,
                "thought_process": [
                    "Intent: Guardrails Fired",
                    "Retrieval Skipped"
                ],
                "status": "Blocked",
                "sources": []
            }

        # ---------- LangGraph ----------
        final_output = rag_agent.invoke(
            initial_state,
            config=config
        )

        return {
            "question": q,
            "answer": final_output.get("final_answer", ""),
            "thought_process": final_output.get("plan", []),
            "status": final_output.get("status", ""),
            "sources": final_output.get("documents", [])
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        logfire.exception(f"Backend Execution Failed: {e}")

        return {
            "question": q,
            "answer": "Internal Server Error",
            "thought_process": [
                "Execution Failed"
            ],
            "status": "error",
            "sources": []
        }