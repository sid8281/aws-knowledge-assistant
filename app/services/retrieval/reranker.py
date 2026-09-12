"""
FlashRank cross-encoder reranker.
"""

from flashrank import Ranker, RerankRequest

from app.config import settings


_ranker = Ranker(
    model_name=settings.RERANKER_MODEL
)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
) -> list[dict]:

    if not candidates:
        return []

    top_k = top_k or settings.RERANK_TOP_K

    passages = [
        {
            "id": index,
            "text": chunk["text"],
        }
        for index, chunk in enumerate(candidates)
    ]

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    ranked = _ranker.rerank(request)

    results = []

    for result in ranked[:top_k]:

        original = candidates[result["id"]]

        results.append({
            **original,
            "rerank_score": float(result["score"]),
        })

    return results