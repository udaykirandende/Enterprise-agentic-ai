import logfire
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Splits text into overlapping character-based chunks.

    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    with logfire.span(
        "Chunking text",
        text_length=len(text),
        chunk_size=chunk_size,
        overlap=overlap,
    ):
        if not text.strip():
            return []

        text = text.strip()
        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - overlap

        return chunks
