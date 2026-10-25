from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """Build an OpenAI-compatible LangChain client routed through Portkey."""

    if not settings.PORTKEY_API_KEY:
        raise RuntimeError("Set PORTKEY_API_KEY to use the Portkey model gateway.")

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1024,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config="pc-rag-fa-7fef4e",
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )
