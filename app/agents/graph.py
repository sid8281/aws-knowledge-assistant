
"""
LangGraph workflow.

Flow:

planner
   │
   ├── conversation ──→ responder
   │
   └── rag ──→ query_rewriter ──→ retriever ──→ grader ──┬── relevant ──────────→ responder
                    ▲                                     │
                    └──────── ambiguous (1 retry) ─────────┤
                                                            └── irrelevant / retry
                                                                exhausted ──→ aws_docs_fallback ──→ responder

Conversation state is persisted using LangGraph checkpointing.

The grader implements a Corrective RAG (CRAG) loop: retrieved chunks
are graded for relevance before the responder ever sees them.
Relevant results proceed as before. Ambiguous results get exactly
one retry through the query rewriter with a broadened search, capped
by state["retry_count"]. Anything still unresolved (irrelevant, or
ambiguous with retries exhausted) falls through to a live AWS
Documentation MCP lookup before the responder gives an honest
"not found" answer -- see aws_docs_fallback.py.
"""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.nodes.planner import plan_node
from app.agents.nodes.query_rewriter import rewrite_query
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.grader import grade_node, route_after_grading
from app.agents.nodes.aws_docs_fallback import aws_docs_fallback_node
from app.agents.nodes.responder import respond_node
from app.agents.memory import memory


class GraphState(TypedDict, total=False):
    question: str
    search_query: str
    intent: str

    retrieved_chunks: list[dict]

    grade: str
    retry_count: int

    answer: str

    steps: list[dict[str, Any]]

    retrieval_stats: dict[str, int]

    sources: list[str]

    # Conversation history
    messages: list[dict[str, str]]


def route_after_planner(state: GraphState) -> str:
    """Route based on planner intent."""

    intent = state.get("intent")

    if intent == "conversation":
        return "conversation"

    if intent == "mcp":
        return "mcp"

    return "rag"


def build_graph():

    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", plan_node)
    graph.add_node("query_rewriter", rewrite_query)
    graph.add_node("retriever", retrieve_node)
    graph.add_node("grader", grade_node)
    graph.add_node("aws_docs_fallback", aws_docs_fallback_node)
    graph.add_node("responder", respond_node)

    # Entry
    graph.set_entry_point("planner")

    # Planner routing
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
           "conversation": "responder",
           "mcp": "aws_docs_fallback",
           "rag": "query_rewriter",
        },
    )

    # RAG path
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "grader")

    # CRAG loop: ambiguous retrieval gets one retry through the
    # rewriter/retriever/grader path; irrelevant (or retry-exhausted
    # ambiguous) falls through to the live AWS Docs MCP fallback
    # before giving up.
    graph.add_conditional_edges(
        "grader",
        route_after_grading,
        {
            "retry": "query_rewriter",
            "aws_docs_fallback": "aws_docs_fallback",
            "responder": "responder",
        },
    )
    graph.add_edge("aws_docs_fallback", "responder")

    # End
    graph.add_edge("responder", END)

    # Persist conversation state
    return graph.compile(checkpointer=memory)


rag_graph = build_graph()


def run_query(
    question: str,
    thread_id: str = "default",
) -> dict:
    """
    Run a question through the workflow.

    Same thread_id:
        Continue the same conversation.

    Different thread_id:
        Start a new conversation.
    """

    initial_state: GraphState = {
        "question": question,
        "steps": [],
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    return rag_graph.invoke(
        initial_state,
        config=config,
    )
