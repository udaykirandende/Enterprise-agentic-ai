"""Local sentence-transformer embeddings shared by ingestion and search."""

import logfire
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 50
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logfire.info("Loading embedding model", model=MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()


def _embed_query(query: str) -> list[float]:
    return _get_model().encode(query, convert_to_numpy=True).tolist()


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if not batch:
        return []
    return _get_model().encode(batch, convert_to_numpy=True).tolist()


def embed_texts(text: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings: list[list[float]] = []
    for start in range(0, len(text), BATCH_SIZE):
        batch = text[start : start + BATCH_SIZE]
        with logfire.span("Embedding batch", model=MODEL_NAME, start=start, size=len(batch)):
            embeddings.extend(model.encode(batch, convert_to_numpy=True).tolist())
    return embeddings
