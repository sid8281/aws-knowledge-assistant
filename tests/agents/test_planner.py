from app.agents.nodes import planner


def test_greeting_is_conversation():
    assert planner.is_conversation("hi") is True
    assert planner.is_conversation("  Hello  ") is True


def test_identity_question_is_conversation():
    assert planner.is_conversation("who are you?") is True


def test_long_aws_question_is_not_conversation_without_llm_call(mocker):
    # A long question skips the LLM classification fallback entirely --
    # this should route to "rag" without ever calling generate_answer.
    mock_generate = mocker.patch.object(planner, "generate_answer")

    result = planner.is_conversation(
        "How did Zomato use Graviton2-based instances to reduce their "
        "infrastructure costs across their order processing pipeline?"
    )

    assert result is False
    mock_generate.assert_not_called()


def test_short_unmatched_phrase_falls_back_to_llm_classification(mocker):
    mock_generate = mocker.patch.object(
        planner, "generate_answer", return_value="conversation"
    )

    result = planner.is_conversation("yo")

    assert result is True
    mock_generate.assert_called_once()


def test_short_aws_question_uses_llm_and_returns_rag(mocker):
    mocker.patch.object(planner, "generate_answer", return_value="rag")

    result = planner.is_conversation("What is S3?")

    assert result is False


def test_llm_classification_failure_defaults_to_rag(mocker):
    mocker.patch.object(
        planner, "generate_answer", side_effect=Exception("network error")
    )

    result = planner.is_conversation("yo")

    assert result is False


def test_plan_node_sets_intent_and_search_query(mocker):
    mocker.patch.object(planner, "generate_answer", return_value="conversation")

    state = {"question": "hello"}

    result = planner.plan_node(state)

    assert result["intent"] == "conversation"
    assert result["search_query"] == ""
    assert result["steps"][-1]["node"] == "planner"


def test_plan_node_routes_rag_for_real_question(mocker):
    mock_generate = mocker.patch.object(planner, "generate_answer")

    state = {
        "question": "What is Amazon S3 and how does its storage "
        "class tiering work for infrequently accessed data?"
    }

    result = planner.plan_node(state)

    assert result["intent"] == "rag"
    assert result["search_query"] == state["question"]
    mock_generate.assert_not_called()
