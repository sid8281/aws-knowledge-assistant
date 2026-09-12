"""
Client for the AWS Documentation MCP server
(https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server).

Each call spawns the server fresh over stdio via `uvx` and tears it
down afterward -- simplest correct implementation, at the cost of a
few seconds of latency on the very first call (uv/uvx downloads and
caches the package; subsequent calls reuse that cache). This is
acceptable here because it's only used as an occasional fallback
when local retrieval comes up empty, not on the hot path of every
request. A high-traffic deployment would want a long-lived
session instead -- see the note in aws_docs_fallback_node.

Requires the `uv`/`uvx` CLI on PATH -- not just the `mcp` Python
package. The Dockerfile installs it; a local dev environment running
`uvicorn` directly needs it installed separately (`pip install uv` or
https://docs.astral.sh/uv/getting-started/installation/). If it's
missing, spawning the subprocess raises FileNotFoundError, which
surfaces through the exceptions below like any other MCP failure.
"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import settings


# Per-call ceiling on the whole MCP round trip (subprocess spawn +
# handshake + tool call). Without this, a hung `uvx` process (slow
# package resolution, a stalled network call inside the MCP server)
# blocks the request indefinitely instead of failing fast into the
# "no results" fallback path.
_MCP_CALL_TIMEOUT_SECONDS = 20


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uvx",
        args=["awslabs.aws-documentation-mcp-server@latest"],
        env={
            "FASTMCP_LOG_LEVEL": "ERROR",
            "AWS_DOCUMENTATION_PARTITION": settings.AWS_DOCS_PARTITION,
        },
    )


def _normalize_search_response(parsed) -> list[dict]:
    """
    The AWS Documentation MCP server's search_documentation return
    shape has drifted across versions -- some versions return a bare
    list[dict], others return a SearchResponse-shaped dict with the
    actual results under a "results" (or "search_results") key plus
    response-level metadata alongside them. Handle both, and always
    return a flat list of dicts so callers never have to guess.
    """

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = (
            parsed.get("results")
            or parsed.get("search_results")
            or []
        )
    else:
        items = []

    return [item for item in items if isinstance(item, dict)]


async def _search_documentation_async(
    search_phrase: str,
    limit: int = 3,
) -> list[dict]:

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "search_documentation",
                {"search_phrase": search_phrase, "limit": limit},
            )

            for content in result.content:
                text = getattr(content, "text", None)
                if text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    return _normalize_search_response(parsed)

            return []


async def _read_documentation_async(url: str) -> str:

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "read_documentation",
                {"url": url},
            )

            for content in result.content:
                text = getattr(content, "text", None)
                if text:
                    return text

            return ""


def search_aws_docs(search_phrase: str, limit: int = 3) -> list[dict]:
    """
    Search live AWS documentation.

    Returns a list of {"url": ..., "title": ..., "context": ...}
    dicts (whatever the MCP server's search API returns) or an empty
    list on any failure -- callers should treat "no results" and
    "the tool failed" the same way (fall back to an honest
    "not found" answer).

    Raises on failure (timeout, missing `uvx`, MCP protocol error)
    instead of swallowing it -- the caller (aws_docs_fallback_node)
    decides how to log/handle that; hiding it here would make every
    failure mode look identical to "AWS docs genuinely had nothing".
    """

    return asyncio.run(
        asyncio.wait_for(
            _search_documentation_async(search_phrase, limit),
            timeout=_MCP_CALL_TIMEOUT_SECONDS,
        )
    )


def read_aws_doc(url: str) -> str:
    """Fetch a specific AWS documentation page as markdown."""

    return asyncio.run(
        asyncio.wait_for(
            _read_documentation_async(url),
            timeout=_MCP_CALL_TIMEOUT_SECONDS,
        )
    )
