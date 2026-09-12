"""
Corrective RAG (CRAG) relevance grader.

Runs after retrieval and before the responder. Grades the reranked
chunks against the search query as one of:

- "relevant"   -> proceed straight to the responder as normal
- "ambiguous"  -> retry once through the query rewriter with a hint
                  to broaden/rephrase, then retrieve again
- "irrelevant" -> skip straight to the responder, but flagged so it
                  gives an honest "not in the knowledge base" answer
                  instead of trying to answer from bad context

This directly closes the gap where retrieval returning weak chunks
used to be treated the same as retrieval returning good chunks --
the responder had no way to tell the difference.

The retry is capped at one attempt (state["retry_count"]) so an
unanswerable question can't loop forever.
"""

import logfire

from app.services.llm import generate_answer


MAX_RETRIES = 1

_GRADE_PROMPT = """You are grading whether retrieved passages are \
relevant enough to answer a search query, for an AWS case-study \
knowledge base.

Search query:
"{query}"

Retrieved passages:
{context}

Grade the passages as ONE of:
- "relevant"   -- at least one passage directly helps answer the query
- "ambiguous"  -- passages are on a related topic but don't clearly \
answer the query; a differently-phrased search might do better
- "irrelevant" -- passages are unrelated to the query

Answer with ONLY one word: relevant, ambiguous, or irrelevant"""


def _build_grading_context(chunks: list[dict]) -> str:
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        # Keep the grading prompt cheap -- a short excerpt is enough
        # to judge topical relevance.
        blocks.append(f"[Passage {i}]\n{text[:500]}")

    return "\n\n".join(blocks)


def grade_node(state: dict) -> dict:

    with logfire.span("rag.grader"):

        chunks = state.get("retrieved_chunks", [])
        query = state.get("search_query", state["question"])

        if not chunks:
            grade = "irrelevant"

            logfire.info(
                "Grading skipped: no retrieved chunks",
                grade=grade,
            )

        else:
            context = _build_grading_context(chunks)

            try:
                raw = generate_answer(
                    _GRADE_PROMPT.format(query=query, context=context)
                )
                grade = raw.strip().lower()
            except Exception as exc:
                # Fail open toward "relevant" -- if grading itself
                # breaks, we fall back to the pre-CRAG behavior
                # (answer from whatever was retrieved) rather than
                # blocking every query on a broken grader call.
                logfire.error(
                    "Grading LLM call failed, defaulting to relevant",
                    error=str(exc),
                )
                grade = "relevant"

            if grade not in ("relevant", "ambiguous", "irrelevant"):
                logfire.warning(
                    "Unexpected grade output, defaulting to relevant",
                    raw_output=raw,
                )
                grade = "relevant"

            logfire.info(
                "Grading completed",
                grade=grade,
                chunk_count=len(chunks),
            )

        state["grade"] = grade

        # Track retries here (not in the conditional edge function,
        # which is read-only in LangGraph) so route_after_grading's
        # MAX_RETRIES cap actually has something to check.
        if grade == "ambiguous" and state.get("retry_count", 0) < MAX_RETRIES:
            state["retry_count"] = state.get("retry_count", 0) + 1

        state["steps"] = state.get("steps", []) + [
            {
                "node": "grader",
                "output": grade,
            }
        ]

        return state


def route_after_grading(state: dict) -> str:
    """
    Decide where to go after grading.

    - "ambiguous" with a retry still available -> one more pass
      through the query rewriter (broadened search).
    - "irrelevant", or "ambiguous" with retries exhausted -> the live
      AWS Docs MCP fallback gets one shot before giving up.
    - "relevant" -> straight to the responder.
    """

    grade = state.get("grade", "relevant")
    retry_count = state.get("retry_count", 0)

    if grade == "ambiguous" and retry_count < MAX_RETRIES:
        return "retry"

    if grade == "irrelevant" or grade == "ambiguous":
        return "aws_docs_fallback"

    return "responder"
