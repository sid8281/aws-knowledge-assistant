# AWS Knowledge Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions
about AWS case studies and technical blog posts. Hybrid retrieval,
LLM-based relevance grading with a corrective retry loop (CRAG),
guardrails, and a Streamlit chat UI, served behind a FastAPI backend.

## Architecture

```
                         ┌─────────────┐
                         │   Planner   │  (conversation vs. RAG intent)
                         └──────┬──────┘
                    conversation│rag
                ┌───────────────┴────────────────┐
                ▼                                 ▼
          ┌───────────┐                  ┌─────────────────┐
          │ Responder │◄────┬────────────┤  Query Rewriter  │◄────┐
          └───────────┘     │            └────────┬─────────┘     │
                ▲            \                     ▼               │
                │             \            ┌───────────────┐       │
                │              \           │   Retriever   │       │
                │               \          │ (dense+BM25+  │       │
                │                \         │  RRF+rerank)  │       │
                │                 \        └───────┬───────┘       │
                │                  \                ▼               │
                │           ┌───────────────┐┌───────────────┐     │
                │           │  AWS Docs MCP ││    Grader     │      │
                │           │   fallback    │◄┤  (CRAG check) │     │
                │           └───────────────┘ └───────┬───────┘     │
                │            irrelevant /        ambiguous          │
                └────────────retry exhausted─┐  (max 1 retry)       │
                                              └──────────────────────┘
```

- **Planner** — routes small talk directly to the responder; real
  questions go through the RAG path. A small phrase list handles the
  obvious cases for free; anything short and unmatched falls back to
  a cheap LLM classification instead of defaulting straight to RAG.
- **Query Rewriter** — turns follow-up questions ("what about it?")
  into standalone search queries using conversation history. On a
  CRAG retry it instead broadens/rephrases the previous query.
- **Retriever** — hybrid search: dense vectors from Qdrant + BM25
  keyword search, combined with Reciprocal Rank Fusion, then
  reranked with FlashRank.
- **Grader (CRAG)** — grades the reranked chunks as relevant,
  ambiguous, or irrelevant before the responder ever sees them.
  Ambiguous results get exactly one retry through the rewriter with
  a broadened query.
- **AWS Docs MCP fallback** — if local retrieval still comes up
  empty (irrelevant, or ambiguous with retries exhausted), queries
  the live [AWS Documentation MCP server](https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server)
  instead of just answering "not found" — the local corpus is a
  fixed snapshot, but AWS's own docs are always current.
- **Responder** — generates the final answer from the graded
  context, or a plain conversational reply.
- **Guardrails** (NeMo Guardrails) — an input check before the
  pipeline runs and an output check before the answer is returned,
  both scoped to AWS/case-study topics.
- **LLM Gateway** (Portkey) — all generation calls go through
  Portkey with an automatic primary → fallback model chain.
- **Observability** (Logfire) — traces every node, LLM call, and
  guardrail decision.

## Project layout

```
app/
  agents/            LangGraph workflow, nodes, conversation memory
  gateway/            Portkey LLM gateway
  guardrails/          NeMo Guardrails config + wrapper
  ingestion/           document loaders + chunking + embed/upsert pipeline
  services/            LLM client, hybrid retrieval (dense/BM25/rerank)
  security.py           API key auth dependency
  main.py                FastAPI app
ui/
  streamlit_app.py     chat UI
tests/                pytest suite (mocked external calls)
evals/                 RAG-quality evaluation (ragas/deepeval)
```

## Setup

### 1. Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (for Qdrant/Postgres, or to run the whole stack)

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

| Variable | Required | Notes |
|---|---|---|
| `API_KEY` | Recommended | Shared secret required in the `X-API-Key` header. Leave empty to disable auth for local dev. |
| `PORTKEY_API_KEY`, `PORTKEY_VIRTUAL_KEY_PRIMARY`, `PORTKEY_VIRTUAL_KEY_FALLBACK` | Yes | LLM access via [Portkey](https://portkey.ai). |
| `OPENAI_API_KEY` | Yes | Used by NeMo Guardrails' self-check LLM calls. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Yes | Vector store connection. |
| `CHECKPOINT_DB_URL` | Recommended | Postgres connection string for conversation memory. Falls back to in-process memory (dev only) if unset. |
| `EMBEDDING_PROVIDER` | No | `local` / `bedrock` / `openai` — see below. Defaults to `local`. |
| `AWS_DOCS_MCP_ENABLED` | No | Live AWS docs fallback when local retrieval finds nothing. Defaults to `true`. |
| `LANGSMITH_API_KEY` | Only for evals | Not required to run the app itself. |

### Choosing an embedding provider (and why it matters for deployment)

`EMBEDDING_PROVIDER` picks how chunk/query embeddings are generated:

| Provider | Where it runs | RAM cost | Setup |
|---|---|---|---|
| `local` (default) | In-process, `sentence-transformers` | High — loads a ~440MB model plus torch itself; can exceed 1GB resident | None (no API key), but requires `uv sync --extra local-embeddings` |
| `bedrock` | Amazon Bedrock Titan Embeddings, via `boto3` | Tiny — just an HTTP call | AWS credentials with Bedrock access |
| `openai` | OpenAI embeddings API | Tiny — just an HTTP call | Reuses `OPENAI_API_KEY` (already needed for guardrails) |

**If you're deploying to a memory-capped host (e.g. Render's free
tier, 512MB RAM) and the app is failing/OOMing on startup while
loading the embedding model, this is why: `local` loads the entire
model into the container's RAM, which alone can exceed the whole
memory budget.** Switch to `bedrock` or `openai`:

```bash
EMBEDDING_PROVIDER=openai   # or bedrock
```

The `local-embeddings` extra (which pulls in `sentence-transformers`
and `torch`) is **not** installed by default for exactly this reason
— see `pyproject.toml` and the `INCLUDE_LOCAL_EMBEDDINGS` build arg
in the `Dockerfile`. Leave it out entirely for a memory-capped
deploy; only set `INCLUDE_LOCAL_EMBEDDINGS=true` (or run
`uv sync --extra local-embeddings` locally) if you're self-hosting
somewhere with more RAM and want to avoid embedding API costs.

**Switching providers requires re-ingesting your documents** into a
fresh or renamed `QDRANT_COLLECTION` — embedding spaces from
different models aren't compatible with each other.

### AWS Documentation MCP fallback

When local retrieval (hybrid search + CRAG grading) comes up empty,
the graph queries the live [AWS Documentation MCP server](https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server)
instead of just answering "I couldn't find that." This needs `uv`/`uvx`
on `PATH` (already true in the provided `Dockerfile`) and outbound
network access to PyPI (first call only — `uvx` caches the package
after that) and `docs.aws.amazon.com`. Disable it with
`AWS_DOCS_MCP_ENABLED=false` if you'd rather it just give the honest
"not found" answer.

### 3. Run everything with Docker Compose (recommended)

```bash
docker compose up --build
```

This starts Qdrant, Postgres, the FastAPI API (`:8000`), and the
Streamlit UI (`:8501`).

### 4. Or run locally

```bash
uv sync

# Start Qdrant + Postgres only
docker compose up qdrant postgres -d

# Ingest documents into the knowledge base (put source files in DATA/)
uv run python -m app.ingestion.loaders.ingest

# Run the API
uv run uvicorn app.main:app --reload

# In a second terminal, run the UI
uv run streamlit run ui/streamlit_app.py
```

## API

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "How did Zomato use AWS Graviton2 instances?"}'
```

Request body:

| Field | Type | Notes |
|---|---|---|
| `question` | string, 1–2000 chars | required |
| `thread_id` | string | optional — omit to start a new conversation; the response returns the id to continue it |

Response:

```json
{
  "answer": "...",
  "sources": ["zomato_case_study.json"],
  "steps": [{"node": "planner", "output": "rag"}, ...],
  "thread_id": "generated-or-supplied-id",
  "blocked": false
}
```

### `GET /health`

Reports per-subsystem status (Qdrant connectivity, BM25 index,
checkpointing backend, auth status) — not just process liveness.

## Testing

```bash
uv run pytest
```

All external calls (LLM, Qdrant, guardrails) are mocked — the suite
runs without any real API keys or network access. CI (`.github/workflows/ci.yml`)
runs lint + tests + a Docker build on every push/PR.

## Evaluation

`evals/` contains a separate RAG-quality evaluation harness (ragas +
deepeval) against a fixed question set in `evals/questions.json`:

```bash
uv run python -m evals.evaluation
```

This is not part of CI — run it manually when you change the
retrieval pipeline or prompts, to check for quality regressions.

### LangSmith evaluation

For versioned, dashboard-tracked evaluation runs (rather than a local
JSON summary):

```bash
uv run python -m evals.langsmith_eval
```

Requires `LANGSMITH_API_KEY` in `.env` (get one at
[smith.langchain.com](https://smith.langchain.com)). This creates/reuses
a LangSmith dataset from the same question set as `evals/evaluation.py`,
runs the full graph against it, and scores each result with:
keyword coverage, source coverage, and an LLM-judged faithfulness
check (using the same Portkey LLM gateway the app uses — no extra
API key needed beyond what's already configured). Results and
per-question traces show up as a new experiment in the LangSmith UI.

## Deployment notes

- Set `API_KEY` and `CHECKPOINT_DB_URL` before deploying anywhere
  reachable from outside your machine — without them, auth is
  disabled and conversation state won't survive a restart or scale
  past one process.
- `RATE_LIMIT` (default `20/minute`) is per-client-IP; adjust for
  your expected traffic.
- The BM25 index and embedding model (if using `EMBEDDING_PROVIDER=local`)
  load at process startup — a new document requires re-running
  ingestion and restarting the API process (or redeploying) to pick
  up the change.
- **On a memory-capped host (Render free tier, etc.), set
  `EMBEDDING_PROVIDER=bedrock` or `openai`** — see "Choosing an
  embedding provider" above. This is the single most common reason
  a deploy OOMs on startup.
- The AWS Docs MCP fallback spawns a subprocess per call — fine for
  occasional fallback use, but a high-traffic deployment would want
  a persistent MCP session instead of reconnecting every time (see
  the note in `app/services/aws_docs_mcp.py`).
