import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "openai/gpt-oss-20b")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("QDRANT_cluster_endpoint")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "uday")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("GROQ_SLUG")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2")

    def validate_api_runtime(self) -> None:
        required = {
            "GROQ_API_KEY": self.GROQ_API_KEY,
            "PORTKEY_API_KEY": self.PORTKEY_API_KEY,
            "GROQ_SLUG": self.GROQ_SLUG,
            "GROQ_SLUG_2": self.GROQ_SLUG_2,
            "QDRANT_URL": self.QDRANT_URL,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )


settings = Settings()
