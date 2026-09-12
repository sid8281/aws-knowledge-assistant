"""
Ingestion entrypoint.

Walks DATA/, loads supported documents,
chunks them, embeds only new chunks,
and upserts them into Qdrant.

Run:
    uv run python -m app.ingestion.loaders.ingest
"""

import hashlib
import os
import json

from app.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.loaders.txt_loader import load_csv_rows
from app.ingestion.chunking import chunk_text
from app.services.retrieval.embeddings import embed_texts
from app.services.retrieval.vector_store import (
    get_client,
    ensure_collection,
    existing_ids,
    upsert_chunks,
)


SUPPORTED_EXTS = {
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".docx",
    ".pptx",
    ".csv",
}


def gather_source_documents(data_dir: str) -> list[dict]:
    """Load all supported documents from DATA/."""

    documents = []

    for root, _, files in os.walk(data_dir):

        for fname in files:

            ext = os.path.splitext(fname)[1].lower()
            path = os.path.join(root, fname)

            if ext not in SUPPORTED_EXTS:
                continue

            print(f"Loading: {path}")

            if ext == ".csv":

                documents.extend(
                    load_csv_rows(path)
                )

            else:

                text = load_document(path)

                if not text.strip():
                    continue

                documents.append(
                    {
                        "source": path,
                        "text": text,
                    }
                )

    return documents


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Create a stable ID for a chunk."""

    value = f"{source}:{chunk_index}"

    return hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()


def process_documents(
    documents: list[dict],
    processed_dir: str
) -> list[dict]:

    """Chunk documents and create stable chunk IDs."""

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    all_chunks = []

    for doc in documents:

        chunks = chunk_text(
            doc["text"]
        )

        doc_chunks = []

        for i, chunk in enumerate(chunks):

            chunk_record = {
                "id": make_chunk_id(
                    doc["source"],
                    i
                ),
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk,
            }

            doc_chunks.append(
                chunk_record
            )

            all_chunks.append(
                chunk_record
            )

        safe_name = "".join(
            c if c.isalnum() else "_"
            for c in os.path.basename(
                doc["source"]
            )
        )[:60]

        out_path = os.path.join(
            processed_dir,
            f"{safe_name}.json"
        )

        with open(
            out_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                doc_chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

    return all_chunks


def main():

    print(
        f"Scanning '{settings.DATA_DIR}' for documents..."
    )

    # --------------------------------
    # Load documents
    # --------------------------------

    documents = gather_source_documents(
        settings.DATA_DIR
    )

    print(
        f"Found {len(documents)} source document(s)."
    )

    if not documents:
        print("No documents found.")
        return

    # --------------------------------
    # Chunk documents
    # --------------------------------

    print("Chunking documents...")

    chunks = process_documents(
        documents,
        settings.PROCESSED_DATA_DIR
    )

    print(
        f"Produced {len(chunks)} chunk(s)."
    )

    if not chunks:
        print("No chunks produced.")
        return

    # --------------------------------
    # Qdrant
    # --------------------------------

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    # --------------------------------
    # First ingestion
    # --------------------------------

    if settings.QDRANT_COLLECTION not in collections:

        print(
            "Qdrant collection does not exist."
        )

        print(
            "Generating initial embeddings..."
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = embed_texts(
            texts
        )

        if not vectors:
            print("No embeddings generated.")
            return

        ensure_collection(
            client,
            vector_size=len(vectors[0])
        )

        upsert_chunks(
            client,
            chunks,
            vectors
        )

        print(
            f"Initial ingestion complete: "
            f"{len(chunks)} chunks."
        )

        return

    # --------------------------------
    # Incremental ingestion
    # --------------------------------

    print(
        "Checking which chunks already exist in Qdrant..."
    )

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    existing = existing_ids(
        client,
        ids
    )

    new_chunks = [
        chunk
        for chunk in chunks
        if chunk["id"] not in existing
    ]

    print(
        f"Existing chunks: {len(existing)}"
    )

    print(
        f"New chunks: {len(new_chunks)}"
    )

    if not new_chunks:

        print(
            "No new chunks to ingest."
        )

        return

    # --------------------------------
    # Embed only new chunks
    # --------------------------------

    print(
        f"Generating embeddings for "
        f"{len(new_chunks)} new chunks..."
    )

    texts = [
        chunk["text"]
        for chunk in new_chunks
    ]

    vectors = embed_texts(
        texts
    )

    # --------------------------------
    # Upsert
    # --------------------------------

    print(
        f"Adding {len(new_chunks)} "
        f"new vectors to Qdrant..."
    )

    upsert_chunks(
        client,
        new_chunks,
        vectors
    )

    print(
        "Incremental ingestion complete."
    )


if __name__ == "__main__":
    main()