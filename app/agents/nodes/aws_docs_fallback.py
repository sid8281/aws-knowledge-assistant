"""
Live AWS Documentation fallback node.

Runs only when the CRAG grader has nothing useful from the local
knowledge base: either the grade is "irrelevant", or it stayed
"ambiguous" after the one allowed retry. The local corpus is a fixed
snapshot of case studies/blogs, but AWS's own documentation is always
current -- so instead of just telling the user "not found", this
queries the live AWS Documentation MCP server
(https://github.com/awslabs/mcp) and, if it finds anything, feeds
those results into the responder like any other retrieved chunk.

If AWS_DOCS_MCP_ENABLED is off, or the MCP call fails or returns
nothing, this is a no-op and the responder falls through to its
existing "couldn't find relevant information" answer -- unchanged
from before this fallback existed.

Note on cost: each call here currently spawns a fresh `uvx` subprocess
(see app/services/aws_docs_mcp.py). That's fine for occasional
fallback use; a high-QPS deployment would want to hold one
long-lived MCP session open instead of reconnecting per request.
"""

import logfire

from app.config import settings
from app.services.aws_docs_mcp import search_aws_docs


def aws_docs_fallback_node(state: dict) -> dict:

    with logfire.span("rag.aws_docs_fallback"):

        if not settings.AWS_DOCS_MCP_ENABLED:
            state["steps"] = state.get("steps", []) + [
                {"node": "aws_docs_fallback", "output": "disabled"}
            ]
            return state

        query = state.get("search_query") or state["question"]

        try:
            results = search_aws_docs(query, limit=3)
        except Exception as exc:
            # Log with a full traceback (not just str(exc)) so the
            # actual cause -- missing `uvx` on PATH, a timeout, a
            # changed MCP response shape, a network failure -- shows
            # up in the logs instead of being indistinguishable from
            # "AWS docs genuinely had nothing for this query".
            logfire.exception(
                "AWS Docs MCP fallback failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            results = []

            state["steps"] = state.get("steps", []) + [
                {
                    "node": "aws_docs_fallback",
                    "output": (
                        f"MCP call failed: {type(exc).__name__}: {exc}"
                    ),
                }
            ]

            return state

        if results:

            chunks = []

            for i, result in enumerate(results):
                text = (
                    result.get("context")
                    or result.get("title")
                    or ""
                )
                url = result.get("url", "")

                if not text or not url:
                    continue

                chunks.append({
                    "id": f"aws-docs-{i}",
                    "text": text,
                    "source": url,
                    "chunk_index": 0,
                })

            if chunks:
                state["retrieved_chunks"] = chunks
                # Let the responder treat this like a normal
                # successful retrieval.
                state["grade"] = "relevant"

                logfire.info(
                    "AWS Docs MCP fallback found results",
                    result_count=len(chunks),
                )

                state["steps"] = state.get("steps", []) + [
                    {
                        "node": "aws_docs_fallback",
                        "output": f"Found {len(chunks)} live AWS doc result(s)",
                    }
                ]

                return state

        logfire.info("AWS Docs MCP fallback found nothing")

        state["steps"] = state.get("steps", []) + [
            {"node": "aws_docs_fallback", "output": "No live AWS doc results"}
        ]

        return state
