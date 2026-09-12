"""
FastAPI entrypoint.

Flow for POST /query:
1. Guardrails input check — block off-topic / unsafe questions.
2. Run the LangGraph pipeline (Planner -> Retriever -> Responder).
3. Guardrails output check — block unsafe / off-policy answers.
4. Return the answer + sources + step trace.
"""

import uuid
from contextlib import asynccontextmanager

import logfire

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.graph import run_query
from app.agents.memory import close_checkpointer
from app.config import settings
from app.guardrails.guard import check_input, check_output
from app.observability import setup_observability
from app.security import verify_api_key


# ---------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy singletons (embedding model, reranker, BM25 index,
    # guardrails config, checkpointer) are all initialized at import
    # time via module-level code in their respective modules. Importing
    # app.agents.graph here forces that initialization to happen during
    # startup rather than lazily on the first request, so a broken
    # dependency (e.g. Qdrant unreachable) fails fast at boot instead of
    # surfacing as a confusing 500 on someone's first query.
    logfire.info("Application startup: dependencies initialized eagerly at import time.")

    yield

    close_checkpointer()
    logfire.info("Application shutdown: checkpointer closed.")


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="AWS Case Studies RAG API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

setup_observability(app)


# ---------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Client-supplied conversation id. Omit to start a new "
            "conversation — the server will generate one and return it "
            "in the response so the client can continue the thread."
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    steps: list[dict]
    thread_id: str
    blocked: bool = False


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness/readiness probe. Reports whether the core subsystems
    that were initialized at startup are actually usable, not just
    that the HTTP server is up.
    """

    checks: dict[str, str] = {}

    try:
        from app.services.retrieval.vector_store import get_client

        get_client().get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    try:
        from app.agents.nodes.retriever import _bm25

        checks["bm25_index"] = (
            "ok" if _bm25.chunks else "error: empty index"
        )
    except Exception as exc:
        checks["bm25_index"] = f"error: {exc}"

    checks["checkpointing"] = (
        "postgres" if settings.CHECKPOINT_DB_URL else "in-memory (dev only)"
    )
    checks["auth"] = "enabled" if settings.API_KEY else "disabled"

    overall_ok = all(
        value == "ok" or value.startswith(("postgres", "in-memory", "enabled", "disabled"))
        for value in checks.values()
    )

    status_code = 200 if overall_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if overall_ok else "degraded", "checks": checks},
    )


# ---------------------------------------------------------
# Query
# ---------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.RATE_LIMIT)
def query(request: Request, body: QueryRequest):

    thread_id = body.thread_id or str(uuid.uuid4())

    with logfire.span(
        "chat.request",
        thread_id=thread_id,
    ):

        logfire.info(
            "Chat request started",
            question_length=len(body.question),
            thread_id=thread_id,
        )

        # -------------------------------------------------
        # 1. Input guardrail
        # -------------------------------------------------

        try:
            allowed, refusal = check_input(body.question)
        except Exception as exc:
            logfire.error("Input guardrail failed", error=str(exc))
            raise HTTPException(
                status_code=502,
                detail="Guardrail service unavailable. Please try again shortly.",
            ) from exc

        if not allowed:

            logfire.info(
                "Request blocked by input guardrail",
                decision="block",
            )

            return QueryResponse(
                answer=refusal,
                sources=[],
                steps=[],
                thread_id=thread_id,
                blocked=True,
            )

        # -------------------------------------------------
        # 2. LangGraph RAG pipeline
        # -------------------------------------------------

        logfire.info("Running RAG pipeline")

        try:
            result = run_query(body.question, thread_id=thread_id)
        except Exception as exc:
            logfire.error("RAG pipeline failed", error=str(exc))
            raise HTTPException(
                status_code=502,
                detail="The assistant is temporarily unavailable. Please try again shortly.",
            ) from exc

        # -------------------------------------------------
        # 3. Output guardrail
        # -------------------------------------------------

        try:
            allowed, final_answer = check_output(result["answer"])
        except Exception as exc:
            logfire.error("Output guardrail failed", error=str(exc))
            raise HTTPException(
                status_code=502,
                detail="Guardrail service unavailable. Please try again shortly.",
            ) from exc

        if not allowed:

            logfire.info(
                "Response blocked by output guardrail",
                decision="block",
            )

            return QueryResponse(
                answer=final_answer,
                sources=[],
                steps=result["steps"],
                thread_id=thread_id,
                blocked=True,
            )

        # -------------------------------------------------
        # 4. Sources
        # -------------------------------------------------

        sources = sorted(
            {
                chunk["source"]
                for chunk in result.get("retrieved_chunks", [])
            }
        )

        logfire.info(
            "Chat request completed",
            sources_count=len(sources),
            steps_count=len(result["steps"]),
            answer_length=len(final_answer or ""),
        )

        return QueryResponse(
            answer=final_answer,
            sources=sources,
            steps=result["steps"],
            thread_id=thread_id,
            blocked=False,
        )
