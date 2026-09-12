"""
Hybrid retrieval pipeline.

1. Dense retrieval from Qdrant.
2. BM25 keyword retrieval.
3. Reciprocal Rank Fusion.
4. FlashRank reranking.
"""

import logfire

from app.config import settings
from app.services.retrieval.embeddings import embed_query
from app.services.retrieval.vector_store import get_client, search
from app.services.retrieval.reranker import rerank
from app.services.retrieval.bm25 import BM25Retriever


_client = get_client()

_bm25 = BM25Retriever(
    settings.PROCESSED_DATA_DIR
)


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.
    """

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    # Dense results
    for rank, chunk in enumerate(dense_results, start=1):

        chunk_id = str(chunk["id"])

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (k + rank)
        )

        chunks[chunk_id] = chunk

    # BM25 results
    for rank, chunk in enumerate(bm25_results, start=1):

        chunk_id = str(chunk["id"])

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (k + rank)
        )

        chunks[chunk_id] = chunk

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        {
            **chunks[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids
    ]

def retrieve_node(state: dict) -> dict:

    with logfire.span("rag.retrieval"):

        query = state["search_query"]

        logfire.info(
            "Retrieval started",
            query=query,
        )

        # dense retrieval
        query_vector = embed_query(query)

        dense_results = search(
            _client,
            query_vector,
            top_k=settings.DENSE_TOP_K,
        )

        # BM25
        bm25_results = _bm25.search(
            query,
            top_k=settings.BM25_TOP_K,
        )

        # RRF
        candidates = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        candidates = candidates[
            :settings.RERANK_CANDIDATE_K
        ]

        # reranking
        top_chunks = rerank(
            query,
            candidates,
            top_k=settings.RERANK_TOP_K,
        )

        logfire.info(
            "Retrieval completed",
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            candidate_count=len(candidates),
            reranked_count=len(top_chunks),
        )

        state["retrieved_chunks"] = top_chunks

        state["steps"] = state.get("steps", []) + [
            {
                "node": "retriever",
                "output": (
                    f"dense={len(dense_results)}, "
                    f"bm25={len(bm25_results)}, "
                    f"rrf={len(candidates)}, "
                    f"reranked={len(top_chunks)}"
                ),
            }
        ]

        return state