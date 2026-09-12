"""
Centralized configuration.

Every setting is read from environment variables so nothing sensitive is
hard-coded. Copy .env.example to .env and fill in your keys — python-dotenv
loads it automatically when the app starts.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- API auth & rate limiting ---
    API_KEY: str = os.getenv("API_KEY", "")
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "20/minute")

    # --- LLM Gateway (Portkey, with Groq as automatic fallback) ---
    PORTKEY_API_KEY: str = os.getenv("PORTKEY_API_KEY", "")
    PORTKEY_CONFIG_SLUG: str = os.getenv("PORTKEY_CONFIG_SLUG", "")
    PORTKEY_VIRTUAL_KEY_PRIMARY: str = os.getenv("PORTKEY_VIRTUAL_KEY_PRIMARY", "")   # e.g. an OpenAI/Gemini virtual key
    PORTKEY_VIRTUAL_KEY_FALLBACK: str = os.getenv("PORTKEY_VIRTUAL_KEY_FALLBACK", "")  # Groq virtual key
    LLM_MODEL_PRIMARY: str = os.getenv("LLM_MODEL_PRIMARY", "openai/gpt-oss-120b")
    LLM_MODEL_FALLBACK: str = os.getenv("LLM_MODEL_FALLBACK", "openai/gpt-oss-20b")
    LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))

    # --- Embeddings ---
    # See app/services/retrieval/embeddings.py for what each provider
    # needs. "local" needs real RAM (not viable on memory-capped hosts
    # like Render's free tier) -- use "bedrock" or "openai" there.
    # Switching providers requires re-ingesting into a fresh/renamed
    # QDRANT_COLLECTION -- embedding spaces aren't interchangeable.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local | bedrock | openai
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")  # "local" provider only
    BEDROCK_EMBEDDING_MODEL_ID: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")  # "bedrock" provider only

    # --- Conversation checkpointing (LangGraph) ---
    # Postgres connection string for persistent conversation memory.
    # Falls back to an in-memory checkpointer (dev/test only — state is
    # lost on restart and isn't shared across worker processes) when unset.
    CHECKPOINT_DB_URL: str = os.getenv("CHECKPOINT_DB_URL", "")

    # --- Vector store (Qdrant) ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "aws_case_studies")

    # --- Reranking (FlashRank) ---
    DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))

    # Number of unique candidates passed to FlashRank
    RERANK_CANDIDATE_K: int = int(
        os.getenv("RERANK_CANDIDATE_K", "30")
    )

    # Final chunks passed to the responder
    RERANK_TOP_K: int = int(
        os.getenv("RERANK_TOP_K", "5")
    )

    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL",
        "ms-marco-TinyBERT-L-2-v2",
    )
    # --- Chunking ---
    MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "1500"))

    # --- Paths ---
    # Ingestion reads from DATA/true_docs by default. DATA/noisy_docs holds
    # intentionally irrelevant documents, kept separate so you can test
    # retrieval robustness (see docs/07_evaluation.md) without polluting
    # the main knowledge base.
    DATA_DIR: str = os.getenv("DATA_DIR", "DATA")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "processed_data")

    # --- Guardrails ---
    GUARDRAILS_CONFIG_DIR: str = os.getenv("GUARDRAILS_CONFIG_DIR", "app/guardrails/config")
    # NeMo Guardrails' self-check LLM calls read OPENAI_API_KEY directly
    # from the environment per rails.co / config.yml -- it isn't read
    # through this Settings class, but .env.example documents it.

    # --- AWS Documentation MCP fallback ---
    # When the local knowledge base returns nothing relevant, query the
    # live AWS Documentation MCP server (https://github.com/awslabs/mcp)
    # instead of just answering "not found". Requires `uv`/`uvx` on PATH
    # (already true inside the provided Dockerfile) and outbound network
    # access to PyPI (first call) and docs.aws.amazon.com.
    AWS_DOCS_MCP_ENABLED: bool = os.getenv("AWS_DOCS_MCP_ENABLED", "true").lower() == "true"
    AWS_DOCS_PARTITION: str = os.getenv("AWS_DOCS_PARTITION", "aws")  # aws | aws-cn

    # --- LangSmith (RAG evaluation) ---
    # Only used by evals/langsmith_eval.py -- not required to run the app.
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "aws-knowledge-assistant")


settings = Settings()
