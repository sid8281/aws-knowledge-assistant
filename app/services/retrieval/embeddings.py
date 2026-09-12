"""
Embeddings, with a swappable provider.

Why this exists: the original implementation always loaded a local
sentence-transformers BGE model (~440MB of weights, plus torch itself
at several hundred MB more) directly into the app process. That's
fine on a laptop but it's exactly what causes an OOM on a memory-
capped host like Render's free tier (512MB RAM) -- the model alone
can exceed the entire container's budget before the app even
finishes starting up.

EMBEDDING_PROVIDER picks how embeddings are generated:

- "local"   -- sentence-transformers, in-process. No API key needed,
               works offline, but needs real RAM (fine for a local
               dev machine or a host with >=1-2GB, NOT fine on
               Render's free tier). Requires the optional
               `local-embeddings` dependency group.
- "bedrock" -- Amazon Bedrock Titan Text Embeddings via boto3. No
               model loaded into the process -- just an HTTP call --
               so RAM usage stays tiny. Needs AWS credentials with
               Bedrock access (fits naturally here since the rest of
               the project is already AWS-themed).
- "openai"  -- OpenAI's embeddings API. Same tiny-RAM benefit as
               Bedrock. Reuses OPENAI_API_KEY (already required for
               NeMo Guardrails' self-check calls), so it's the
               lowest-setup option if you don't want to touch IAM.

IMPORTANT: embedding spaces are NOT interchangeable. Switching
providers means every existing chunk in Qdrant has vectors in the
wrong space -- re-run ingestion (`uv run python -m
app.ingestion.loaders.ingest`) against a fresh/renamed
QDRANT_COLLECTION after changing this setting.
"""

from app.config import settings


# -------------------------------------------------------------------
# Local (sentence-transformers) -- lazy-loaded so importing this
# module never pulls in torch unless this provider is actually used.
# -------------------------------------------------------------------

_local_model = None


def _get_local_model():
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    return _local_model


# Preload eagerly, but ONLY when "local" is actually the selected
# provider. This restores the original fast-per-request behavior
# (the model loads once at process startup, not on whichever request
# happens to arrive first) while still keeping torch/sentence-
# transformers completely out of the import graph -- and therefore
# out of process RAM -- for "bedrock"/"openai" deployments, which is
# the whole point of the provider split (see the module docstring).
if settings.EMBEDDING_PROVIDER == "local":
    _get_local_model()


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    embeddings = _get_local_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def _embed_query_local(query: str) -> list[float]:
    embedding = _get_local_model().encode(
        query,
        normalize_embeddings=True,
    )

    return embedding.tolist()


# -------------------------------------------------------------------
# Amazon Bedrock (Titan Text Embeddings)
# -------------------------------------------------------------------

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client

    if _bedrock_client is None:
        import boto3

        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
        )

    return _bedrock_client


def _embed_one_bedrock(text: str) -> list[float]:
    import json

    response = _get_bedrock_client().invoke_model(
        modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )

    body = json.loads(response["body"].read())

    return body["embedding"]


def _embed_texts_bedrock(texts: list[str]) -> list[list[float]]:
    # Titan's embedding models take one input per call -- there's no
    # batch endpoint, so this is sequential. Fine for ingestion (a
    # one-off, offline job); if the corpus grows large enough for
    # this to matter, parallelize with a thread pool.
    return [_embed_one_bedrock(text) for text in texts]


def _embed_query_bedrock(query: str) -> list[float]:
    return _embed_one_bedrock(query)


# -------------------------------------------------------------------
# OpenAI
# -------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client

    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    return _openai_client


def _embed_texts_openai(texts: list[str]) -> list[list[float]]:
    response = _get_openai_client().embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def _embed_query_openai(query: str) -> list[float]:
    return _embed_texts_openai([query])[0]


# -------------------------------------------------------------------
# Public interface -- unchanged signatures, so nothing calling into
# this module (retriever.py, ingest.py) needs to change.
# -------------------------------------------------------------------

_PROVIDERS = {
    "local": (_embed_texts_local, _embed_query_local),
    "bedrock": (_embed_texts_bedrock, _embed_query_bedrock),
    "openai": (_embed_texts_openai, _embed_query_openai),
}


def _provider_functions():
    try:
        return _PROVIDERS[settings.EMBEDDING_PROVIDER]
    except KeyError:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            f"Expected one of: {', '.join(_PROVIDERS)}"
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for document chunks (used by ingestion)."""

    embed_texts_fn, _ = _provider_functions()
    return embed_texts_fn(texts)


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a user query (used at retrieval time)."""

    _, embed_query_fn = _provider_functions()
    return embed_query_fn(query)
