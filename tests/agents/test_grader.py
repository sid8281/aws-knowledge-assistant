from app.agents.nodes import grader


def test_grade_node_no_chunks_is_irrelevant():
    state = {"question": "What is Lambda?", "retrieved_chunks": []}

    result = grader.grade_node(state)

    assert result["grade"] == "irrelevant"
    assert result["steps"][-1]["node"] == "grader"


def test_grade_node_relevant(mocker):
    mocker.patch.object(grader, "generate_answer", return_value="relevant")

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "retrieved_chunks": [{"text": "Lambda is a serverless compute service."}],
    }

    result = grader.grade_node(state)

    assert result["grade"] == "relevant"
    assert result.get("retry_count", 0) == 0


def test_grade_node_ambiguous_increments_retry_count(mocker):
    mocker.patch.object(grader, "generate_answer", return_value="ambiguous")

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "retrieved_chunks": [{"text": "unrelated passage"}],
    }

    result = grader.grade_node(state)

    assert result["grade"] == "ambiguous"
    assert result["retry_count"] == 1


def test_grade_node_ambiguous_does_not_exceed_max_retries(mocker):
    mocker.patch.object(grader, "generate_answer", return_value="ambiguous")

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "retrieved_chunks": [{"text": "unrelated passage"}],
        "retry_count": 1,  # already at MAX_RETRIES
    }

    result = grader.grade_node(state)

    assert result["retry_count"] == 1  # unchanged, cap respected


def test_grade_node_llm_failure_defaults_to_relevant(mocker):
    mocker.patch.object(
        grader, "generate_answer", side_effect=Exception("network error")
    )

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "retrieved_chunks": [{"text": "some passage"}],
    }

    result = grader.grade_node(state)

    assert result["grade"] == "relevant"


def test_grade_node_unexpected_output_defaults_to_relevant(mocker):
    mocker.patch.object(grader, "generate_answer", return_value="maybe??")

    state = {
        "question": "What is Lambda?",
        "search_query": "What is Lambda?",
        "retrieved_chunks": [{"text": "some passage"}],
    }

    result = grader.grade_node(state)

    assert result["grade"] == "relevant"


def test_route_after_grading_relevant_goes_to_responder():
    assert grader.route_after_grading({"grade": "relevant"}) == "responder"


def test_route_after_grading_irrelevant_goes_to_aws_docs_fallback():
    assert grader.route_after_grading({"grade": "irrelevant"}) == "aws_docs_fallback"


def test_route_after_grading_ambiguous_retries_once():
    assert (
        grader.route_after_grading({"grade": "ambiguous", "retry_count": 0})
        == "retry"
    )


def test_route_after_grading_ambiguous_falls_to_aws_docs_after_max_retries():
    assert (
        grader.route_after_grading({"grade": "ambiguous", "retry_count": 1})
        == "aws_docs_fallback"
    )
