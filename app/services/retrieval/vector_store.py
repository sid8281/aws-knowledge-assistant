"""Qdrant vector-store operations."""

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import settings


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


def ensure_collection(
    client: QdrantClient,
    vector_size: int,
) -> None:

    collection_names = {
        collection.name
        for collection in client.get_collections().collections
    }

    if settings.QDRANT_COLLECTION in collection_names:
        return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def existing_ids(
    client: QdrantClient,
    ids: list[str],
) -> set[str]:

    if not ids:
        return set()

    normalized_ids = [str(chunk_id) for chunk_id in ids]

    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=normalized_ids,
        with_payload=False,
        with_vectors=False,
    )

    return {
        str(point.id)
        for point in points
    }


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:

    if not chunks:
        return

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunks ({len(chunks)}) and vectors "
            f"({len(vectors)}) must have the same length."
        )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):

        batch_chunks = chunks[start:start + batch_size]
        batch_vectors = vectors[start:start + batch_size]

        points = []

        for chunk, vector in zip(
            batch_chunks,
            batch_vectors,
        ):

            points.append(
                PointStruct(
                    id=str(chunk["id"]),
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
) -> list[dict]:
    """
    NOTE: uses query_points(), not search(). QdrantClient.search() was
    deprecated and then removed in recent qdrant-client releases
    (confirmed gone by 1.19.0) -- query_points() is the current,
    stable replacement and has been available since qdrant-client 1.7.
    """

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )

    chunks = []

    for hit in response.points:

        payload = hit.payload or {}

        if not payload.get("text"):
            continue

        chunks.append({
            "id": str(hit.id),
            "text": payload["text"],
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index"),
            "score": float(hit.score),
        })

    return chunks