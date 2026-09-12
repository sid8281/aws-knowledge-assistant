from app.agents.nodes.retriever import reciprocal_rank_fusion


def test_rrf_combines_and_ranks_by_score():
    dense = [
        {"id": "a", "text": "dense-1"},
        {"id": "b", "text": "dense-2"},
    ]
    bm25 = [
        {"id": "b", "text": "bm25-1"},
        {"id": "c", "text": "bm25-2"},
    ]

    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    fused_ids = [chunk["id"] for chunk in fused]

    # "b" appears in both lists (rank 2 in dense, rank 1 in bm25) so it
    # should score higher than items that only appear once.
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c"}


def test_rrf_empty_inputs_returns_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_preserves_chunk_payload():
    dense = [{"id": "x", "text": "hello", "source": "doc.json"}]

    fused = reciprocal_rank_fusion(dense, [])

    assert fused[0]["text"] == "hello"
    assert fused[0]["source"] == "doc.json"
    assert "rrf_score" in fused[0]
