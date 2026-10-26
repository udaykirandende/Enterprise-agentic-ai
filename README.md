# Enterprise Agentic RAG API

A FastAPI application that routes user requests through a LangGraph workflow for conversational responses, document retrieval, calculator requests, and web search.

## Current capabilities

- **LangGraph routing:** planner selects conversational, retrieval, calculator, or web-search paths.
- **Retrieval-augmented generation:** Qdrant vector search, sentence-transformer embeddings, and FlashRank reranking.
- **Web search:** Tavily search with optional explicit domain restrictions.
- **Calculator:** restricted expression evaluation.
- **Conversation memory:** LangGraph `MemorySaver` keeps state by thread ID while the process is running.
- **Guardrails and observability:** NeMo Guardrails and Logfire instrumentation.
- **API:** FastAPI query endpoint and a workflow graph endpoint.

> **Scope and limitations:** This is a tool-routed LangGraph RAG application, not yet a system of independently collaborating specialist agents. `MemorySaver` is in-memory and does not survive process restarts. The repository contains a six-case evaluation snapshot (6/6 passed, 13.29 s mean latency); it is a small recorded run, not a statistically robust accuracy benchmark or a production-load test. It has not been rerun in this session.

## Architecture

```text
Client
  |
  v
FastAPI /query
  |
  v
Guardrails gate
  |
  v
LangGraph planner
  |---- CONVERSATIONAL ----> Responder ----> Response
  |---- RETRIEVAL ---------> Qdrant search -> FlashRank rerank -> Responder
  |---- WEB_SEARCH --------> Tavily search -> Responder
  |---- CALCULATOR --------> Restricted calculator ------------> Response
```

## Requirements

- Python 3.11
- API credentials for the configured model gateway and Groq guardrail model
- A Qdrant collection containing vectors created with the same embedding model
- Tavily API access for web-search requests
- A Logfire token is optional for local development, but observability will be limited without it

## Local setup (Windows PowerShell)

1. Clone the repository and enter its directory:

   ```powershell
   git clone https://github.com/udaykirandende/Enterprise-agentic-ai.git
   cd Enterprise-agentic-ai
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Fill in the credentials and endpoint values in `.env`. Do not commit `.env` or paste API keys into issues, logs, or chat.

6. Start the API:

   ```powershell
   uvicorn app.main:app --reload
   ```

Open:
- API health/root response: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- Workflow graph image: http://127.0.0.1:8000/graph

## Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key for guardrail model access |
| `GROQ_MODEL` | Configured primary model name |
| `GROQ_GUARD_MODEL` | Model used for guardrail checks |
| `QDRANT_URL` | Qdrant endpoint |
| `QDRANT_API_KEY` | Qdrant API key, if required |
| `QDRANT_COLLECTION` | Vector collection name (default in code: `uday`) |
| `PORTKEY_API_KEY` | Portkey gateway credential |
| `GROQ_SLUG`, `GROQ_SLUG_2` | Required by runtime configuration validation where invoked |
| `LOGFIRE_TOKEN` | Optional Logfire token for telemetry |
| `TAVILY_API_KEY` | Tavily key for web search; required for that feature |

Use the provider's current dashboard to verify model IDs and gateway configuration. Do not assume that example model identifiers or gateway configuration IDs are valid for every account.

## API example

```powershell
$body = @{ q = "Who is Jon Snow?"; thread_id = "demo-session" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType "application/json" -Body $body
```

The response includes the question, answer, status, plan trace, and source documents when available.

## Evaluation snapshot

The checked-in `tests/eval_results.json` records these six cases as passing:

| Case | Type | Recorded latency |
|---|---|---:|
| rag-001 | Retrieval answer from document context | 12.77 s |
| rag-002 | Abstention when the answer is not in context | 7.93 s |
| calc-001 | Calculator: multiplication | 2.76 s |
| calc-002 | Calculator: division | 9.21 s |
| web-001 | Web search with sources | 31.02 s |
| chat-001 | Conversational response | 16.05 s |
| **Overall** | **6/6 cases passed (100% on this small suite)** | **13.29 s mean** |

These are the values in the committed result file, not a fresh run from this session. They should be described as a **six-case evaluation pass rate**, not general model accuracy. Latency varies with provider, network, cold starts, and model routing.

## Tests

Run the unit tests that do not require live provider credentials:

```powershell
pytest -q tests/test_tool_nodes.py
```

The memory tests in `tests/test_memory.py` call a running API. Start the app and run them separately:

```powershell
pytest -q tests/test_memory.py
```

For repeatable retrieval quality evaluation, configure a representative test dataset and record metrics such as retrieval hit rate, answer correctness, citation/source correctness, latency, and token/cost usage. Do not publish unmeasured metrics as project results.

## Docker and Docker Compose

1. Copy `.env.example` to `.env` and fill in the required provider credentials. Keep `.env` private.
2. Create a local `data` folder and put documents to ingest inside it. The ingestion loader supports PDF, HTML, TXT, DOCX, and PPTX files.
3. Start the API and persistent local Qdrant service:

   ```powershell
   docker compose up --build -d
   ```

4. Ingest documents into Qdrant (the command below **recreates the collection**, so use it only when you intend to replace the indexed data):

   ```powershell
   docker compose run --rm api python -m app.ingestion.processor /app/data --wipe
   ```

5. Open http://127.0.0.1:8000/docs. Check the API root at http://127.0.0.1:8000/ and Qdrant at http://127.0.0.1:6333/dashboard.

Stop the stack with `docker compose down`. Persistent Qdrant data remains in the `qdrant_storage` volume; `docker compose down -v` deletes that volume.

For a standalone image build:

```powershell
docker build -t enterprise-agentic-rag .
docker run --rm -p 8000:8000 --env-file .env enterprise-agentic-rag
```

The API container still depends on configured external model-gateway and guardrail providers. Docker Compose supplies Qdrant locally; it does not supply API keys or populate the knowledge base automatically.

## Security notes

- Keep secrets in environment variables or a secret manager.
- Treat retrieved document text and web-search snippets as untrusted input.
- Review guardrail behavior with adversarial test cases; guardrails are not a substitute for authorization or data-access controls.
- The current in-memory conversation checkpointer is not durable and is not intended for multi-worker production deployment.
