"""
Shared fixtures.

Nothing in this test suite makes a real network call — LLM, Qdrant,
embedding, and guardrail calls are all mocked. This keeps CI fast,
free, and deterministic.
"""

import os

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("PORTKEY_API_KEY", "test")
os.environ.setdefault("PORTKEY_VIRTUAL_KEY_PRIMARY", "test")
os.environ.setdefault("PORTKEY_VIRTUAL_KEY_FALLBACK", "test")

import pytest


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "id": "1",
            "text": "AWS Lambda is a serverless compute service.",
            "source": "lambda_case_study.json",
            "chunk_index": 0,
        },
        {
            "id": "2",
            "text": "Qdrant is a vector database used for similarity search.",
            "source": "qdrant_case_study.json",
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def mock_llm_generate(mocker):
    """Patch app.gateway.portkey_gateway.generate to avoid real API calls."""

    return mocker.patch(
        "app.gateway.portkey_gateway.generate",
        return_value="Mocked LLM response.",
    )
