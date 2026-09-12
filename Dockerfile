# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set to "true" to install sentence-transformers/torch for
# EMBEDDING_PROVIDER=local (needs real RAM -- don't set this for
# Render's free tier or similar memory-capped hosts; use "bedrock"
# or "openai" as EMBEDDING_PROVIDER instead and leave this false).
ARG INCLUDE_LOCAL_EMBEDDINGS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by flashrank / psycopg (and torch, if the
# local-embeddings extra is enabled below) at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, locked installs from pyproject.toml + uv.lock,
# and uvx (used at runtime by the AWS Documentation MCP fallback).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Dependency layer (cached separately from app code) ----
COPY pyproject.toml uv.lock ./
RUN if [ "$INCLUDE_LOCAL_EMBEDDINGS" = "true" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra local-embeddings; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# ---- App layer ----
COPY app ./app
COPY processed_data ./processed_data

# Activate the venv uv created
ENV PATH="/app/.venv/bin:$PATH"

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
