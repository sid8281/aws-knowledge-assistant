import sys

import pytest

from app.services.retrieval import embeddings


def test_unknown_provider_raises(mocker):
    mocker.patch.object(embeddings.settings, "EMBEDDING_PROVIDER", "bogus")

    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        embeddings._provider_functions()


def test_bedrock_provider_resolves(mocker):
    mocker.patch.object(embeddings.settings, "EMBEDDING_PROVIDER", "bedrock")

    embed_texts_fn, embed_query_fn = embeddings._provider_functions()

    assert embed_texts_fn is embeddings._embed_texts_bedrock
    assert embed_query_fn is embeddings._embed_query_bedrock


def test_openai_provider_resolves(mocker):
    mocker.patch.object(embeddings.settings, "EMBEDDING_PROVIDER", "openai")

    embed_texts_fn, embed_query_fn = embeddings._provider_functions()

    assert embed_texts_fn is embeddings._embed_texts_openai
    assert embed_query_fn is embeddings._embed_query_openai


def test_selecting_non_local_provider_never_imports_torch():
    """
    The whole point of the provider split: a memory-capped host (e.g.
    Render's free tier) that selects "bedrock" or "openai" must never
    pull sentence-transformers/torch into the process -- that's the
    hundreds-of-MB of RAM that causes the OOM in the first place.
    """

    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_bedrock_embed_query_calls_titan_api(mocker):
    mocker.patch.object(embeddings.settings, "EMBEDDING_PROVIDER", "bedrock")

    mock_body = mocker.Mock()
    mock_body.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'

    mock_client = mocker.Mock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    mocker.patch.object(embeddings, "_get_bedrock_client", return_value=mock_client)

    result = embeddings.embed_query("What is Lambda?")

    assert result == [0.1, 0.2, 0.3]
    mock_client.invoke_model.assert_called_once()


def test_openai_embed_texts_calls_api(mocker):
    mocker.patch.object(embeddings.settings, "EMBEDDING_PROVIDER", "openai")

    mock_item = mocker.Mock(embedding=[0.4, 0.5])
    mock_response = mocker.Mock(data=[mock_item])

    mock_client = mocker.Mock()
    mock_client.embeddings.create.return_value = mock_response

    mocker.patch.object(embeddings, "_get_openai_client", return_value=mock_client)

    result = embeddings.embed_texts(["What is S3?"])

    assert result == [[0.4, 0.5]]
