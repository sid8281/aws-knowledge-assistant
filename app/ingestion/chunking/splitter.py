"""
Simple paragraph-based text splitter.

Strategy (kept intentionally simple and explainable):
1. Split the document on blank lines -> paragraphs.
2. Greedily pack paragraphs into a chunk until adding the next paragraph
   would exceed MAX_CHUNK_CHARS.
3. If a single paragraph is itself longer than MAX_CHUNK_CHARS, hard-split
   it on character count so no chunk ever exceeds the limit.

This keeps related sentences together (better retrieval quality than a
fixed-size sliding window) while still guaranteeing a predictable max size.
"""

from app.config import settings


def split_into_paragraphs(text: str) -> list[str]:
    raw_paragraphs = text.split("\n\n")
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if p:
            paragraphs.append(p)
    return paragraphs


def chunk_text(text: str, max_chars: int = None) -> list[str]:
    """Turn raw document text into a list of chunks, each <= max_chars."""
    max_chars = max_chars or settings.MAX_CHUNK_CHARS
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current = ""

    for para in paragraphs:
        # Hard-split any paragraph that alone exceeds the limit
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
            continue

        # Would adding this paragraph overflow the current chunk?
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
