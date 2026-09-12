from app.agents.nodes import query_rewriter


def test_no_history_no_retry_uses_question_as_is(mocker):
    mock_generate = mocker.patch.object(query_rewriter, "generate_answer")

    state = {"question": "What is S3?"}

    result = query_rewriter.rewrite_query(state)

    assert result["search_query"] == "What is S3?"
    mock_generate.assert_not_called()


def test_with_history_calls_llm_rewriter(mocker):
    mocker.patch.object(
        query_rewriter, "generate_answer", return_value="What is AWS Lambda?"
    )

    state = {
        "question": "what about it?",
        "messages": [
            {"role": "user", "content": "Tell me about Lambda"},
            {"role": "assistant", "content": "Lambda is serverless compute."},
        ],
    }

    result = query_rewriter.rewrite_query(state)

    assert result["search_query"] == "What is AWS Lambda?"


def test_empty_rewrite_falls_back_to_question(mocker):
    mocker.patch.object(query_rewriter, "generate_answer", return_value="")

    state = {
        "question": "what about it?",
        "messages": [{"role": "user", "content": "Tell me about Lambda"}],
    }

    result = query_rewriter.rewrite_query(state)

    assert result["search_query"] == "what about it?"


def test_crag_retry_uses_broadening_instructions_even_without_history(mocker):
    mock_generate = mocker.patch.object(
        query_rewriter, "generate_answer", return_value="AWS serverless compute pricing"
    )

    state = {
        "question": "What is Lambda pricing?",
        "search_query": "What is Lambda pricing?",
        "grade": "ambiguous",
    }

    result = query_rewriter.rewrite_query(state)

    assert result["search_query"] == "AWS serverless compute pricing"
    prompt_used = mock_generate.call_args[0][0]
    assert "did not retrieve clearly relevant results" in prompt_used
