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

n checkpointer is not durable and is not intended for multi-worker production deployment.
