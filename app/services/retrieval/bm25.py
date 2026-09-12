"""BM25 keyword retrieval."""

import json
import os
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, processed_dir: str):

        self.chunks: list[dict] = []
        self.bm25 = None

        self._load_chunks(processed_dir)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Normalize text into lowercase keyword tokens.
        """

        return re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

    def _load_chunks(
        self,
        processed_dir: str,
    ) -> None:

        for filename in sorted(os.listdir(processed_dir)):

            if not filename.endswith(".json"):
                continue

            path = os.path.join(
                processed_dir,
                filename,
            )

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    chunks = json.load(file)

            except (OSError, json.JSONDecodeError) as exc:

                print(
                    f"Skipping {filename}: {exc}"
                )

                continue

            if not isinstance(chunks, list):
                continue

            self.chunks.extend(chunks)

        if not self.chunks:
            raise ValueError(
                "BM25 index contains no chunks."
            )

        documents = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(documents)

        print(
            f"BM25 indexed {len(self.chunks)} chunks."
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict]:

        if not query.strip():
            return []

        tokens = self.tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index]

            results.append({
                "id": str(chunk["id"]),
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk.get("chunk_index"),
                "score": float(scores[index]),
            })

        return results