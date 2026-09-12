from app.agents.nodes import aws_docs_fallback


def test_disabled_is_noop(mocker):
    mocker.patch.object(aws_docs_fallback.settings, "AWS_DOCS_MCP_ENABLED", False)
    mock_search = mocker.patch.object(aws_docs_fallback, "search_aws_docs")

    state = {"question": "What is Lambda?", "search_query": "What is Lambda?"}

    result = aws_docs_fallback.aws_docs_fallback_node(state)

    assert result["steps"][-1]["output"] == "disabled"
    mock_search.assert_not_called()


def test_results_populate_chunks_and_reset_grade(mocker):
    mocker.patch.object(aws_docs_fallback.settings, "AWS_DOCS_MCP_ENABLED", True)
    mocker.patch.object(
        aws_docs_fallback,
        "search_aws_docs",
        return_value=[
            {
                "url": "https://docs.aws.amazon.com/lambda/",
                "title": "AWS Lambda",
                "context": "AWS Lambda is a serverless compute service.",
            }
        ],
    )

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "grade": "irrelevant",
    }

    result = aws_docs_fallback.aws_docs_fallback_node(state)

    assert result["grade"] == "relevant"
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["source"] == "https://docs.aws.amazon.com/lambda/"


def test_no_results_leaves_state_for_responder_fallback(mocker):
    mocker.patch.object(aws_docs_fallback.settings, "AWS_DOCS_MCP_ENABLED", True)
    mocker.patch.object(aws_docs_fallback, "search_aws_docs", return_value=[])

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "grade": "irrelevant",
    }

    result = aws_docs_fallback.aws_docs_fallback_node(state)

    assert result["grade"] == "irrelevant"
    assert "retrieved_chunks" not in result
    assert result["steps"][-1]["output"] == "No live AWS doc results"


def test_mcp_failure_is_caught_and_treated_as_no_results(mocker):
    mocker.patch.object(aws_docs_fallback.settings, "AWS_DOCS_MCP_ENABLED", True)
    mocker.patch.object(
        aws_docs_fallback,
        "search_aws_docs",
        side_effect=Exception("uvx not found"),
    )

    state = {"question": "What is Lambda?", "search_query": "What is Lambda?"}

    result = aws_docs_fallback.aws_docs_fallback_node(state)

    assert result["steps"][-1]["output"] == "No live AWS doc results"
