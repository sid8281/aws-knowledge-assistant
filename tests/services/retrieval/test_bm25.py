from app.services.retrieval.bm25 import BM25Retriever


def test_tokenize_lowercases_and_splits():
    tokens = BM25Retriever.tokenize("AWS Lambda Functions!")

    assert tokens == ["aws", "lambda", "functions"]


def test_tokenize_handles_numbers_and_underscores():
    tokens = BM25Retriever.tokenize("EC2_instance costs $100")

    assert tokens == ["ec2_instance", "costs", "100"]


def test_tokenize_empty_string():
    assert BM25Retriever.tokenize("") == []
