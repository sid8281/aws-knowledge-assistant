"""
Conversation memory for the LangGraph RAG application.

Uses a Postgres-backed checkpointer when CHECKPOINT_DB_URL is set, so
conversation state survives restarts and is shared across worker
processes. Falls back to an in-memory checkpointer when it isn't —
fine for local dev/tests, but NOT suitable for a real deployment:
state is per-process and lost on restart.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver

from app.config import settings

logger = logging.getLogger(__name__)


def _build_checkpointer():
    if not settings.CHECKPOINT_DB_URL:
        logger.warning(
            "CHECKPOINT_DB_URL is not set — using in-memory "
            "conversation checkpointing. This is fine for local "
            "dev/tests only: state will be lost on restart and is "
            "NOT shared across worker processes. Set CHECKPOINT_DB_URL "
            "before deploying."
        )
        return MemorySaver()

    from langgraph.checkpoint.postgres import PostgresSaver

    # PostgresSaver.from_conn_string() is a context manager. We enter it
    # once and hold it open for the process lifetime, and expose the
    # context manager itself so main.py's lifespan handler can exit it
    # cleanly on shutdown (closes the connection pool).
    global _checkpointer_cm
    _checkpointer_cm = PostgresSaver.from_conn_string(
        settings.CHECKPOINT_DB_URL
    )
    checkpointer = _checkpointer_cm.__enter__()
    checkpointer.setup()

    logger.info("Using Postgres-backed conversation checkpointing.")

    return checkpointer


_checkpointer_cm = None

# One shared checkpointer for the application.
memory = _build_checkpointer()


def close_checkpointer() -> None:
    """Call on app shutdown to release the Postgres connection pool."""

    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)