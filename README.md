# Enterprise Agentic RAG API

A FastAPI application that routes user requests through a LangGraph workflow for conversational responses, document retrieval, calculator requests, and web search.

## Current capabilities

- **LangGraph routing:** planner selects conversational, retrieval, calculator, or web-search paths.
- **Retrieval-augmented generation:** Qdrant vector search, sentence-transformer embeddings, and FlashRank reranking.
- **Web search:** Tavily search with optional explicit domain restrictions.
- **Calculator:** restricted expression evaluation.
- **Conversation memory:** PostgreSQL-backed LangGraph checkpoints when `DATABASE_URL` is configured; `MemorySaver` is a local/test fallback.
- **Guardrails and observability:** NeMo Guardrails and Logfire instrumentation.
- **MCP integration:** Optional MCP server connections with an explicit tool allowlist (`MCP_ALLOWED_TOOLS`).
- **API:** FastAPI query endpoint and a workflow graph endpoint.

> **Scope and limitations:** This is a tool-routed LangGraph application, not a system of independently collaborating specialist agents. Durable memory requires `DATABASE_URL` to point to PostgreSQL; without it, memory is process-local. MCP is optional and only invokes explicitly allowlisted tools. The checked-in six-case evaluation snapshot is historical and is not a fresh test run or a production accuracy benchmark.

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
  |---- MCP_TOOL ----------> Allowlisted MCP tool -----------------> Response
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
| `DATABASE_URL` | PostgreSQL connection string for durable conversation checkpoints |
| `MCP_SERVERS_JSON` | JSON object configuring one or more MCP servers; optional |
| `MCP_ALLOWED_TOOLS` | Comma-separated exact MCP tool names allowed to execute |

Use the provider's current dashboard to verify model IDs and gateway configuration. Do not assume that example model identifiers or gateway configuration IDs are valid for every account.

## Durable conversation memory

Set `DATABASE_URL` to a PostgreSQL database accessible from the API process. On startup, the app initializes the LangGraph PostgreSQL checkpointer; if the database is unreachable or checkpoint setup fails, startup fails instead of silently claiming persistence. Keep the connection string in the host's secret/environment settings, never in source control.

When a request omits `thread_id`, the API generates a new UUID and returns it in the response. Send that returned ID on later requests to continue the same conversation. Clients should use a distinct thread ID per conversation and must not share IDs between users. The API does not implement user authentication, so production deployments should bind conversation IDs to authenticated identities.

Without `DATABASE_URL`, the app uses in-memory `MemorySaver` for local development and tests; state is lost on process restart.

## MCP tools

MCP is optional. Configure `MCP_SERVERS_JSON` with the server configuration format supported by `langchain-mcp-adapters`, and set `MCP_ALLOWED_TOOLS` to the exact comma-separated tool names that may execute. Example for a Streamable HTTP server:

```dotenv
MCP_SERVERS_JSON={"example":{"transport":"streamable_http","url":"https://your-mcp-server.example/mcp"}}
MCP_ALLOWED_TOOLS=approved_tool_name
```

The example URL and tool name are placeholders, not a working server. Do not enable tools that can execute arbitrary code, access secrets, or perform destructive actions without additional authorization. MCP tool calls are selected by the model but checked against the allowlist before execution. MCP requests fail closed when no server or allowed tool is configured. Verify the actual names returned by your server before setting the allowlist.

## API example

```powershell
$body = @{ q = "Who is Jon Snow?"; thread_id = "demo-session" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType "application/json" -Body $body
```

The response includes the question, answer, status, plan trace, and source documents when available.

## Evaluation snapshot

The checked-in `tests/eval_results.json` is a historical snapshot of six cases. To run the current live evaluation, start the API with valid model/Qdrant/Tavily configuration and run `python -m tests.evaluate_rag` (or set `AGENT_API_URL` to a different endpoint). The runner records mean and p95 latency plus request errors. The checked-in snapshot is not updated by CI. It records these six cases as passing:

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

Run the isolated unit tests that do not require live provider credentials:

```powershell
pytest -q tests/test_tool_nodes.py tests/test_mcp_tools.py
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
