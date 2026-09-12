"""
RAG quality evaluation using LangSmith.

Evaluates the AWS Knowledge Assistant using:
1. Keyword Coverage       - Does the answer contain expected information?
2. Source Coverage        - Did retrieval find expected sources?
3. Faithfulness           - Is the answer supported by retrieved context?
4. Answer Correctness     - Is the final answer correct and useful?

All metrics are reported on a 1-10 scale.

Setup:
    Set LANGSMITH_API_KEY in .env.

Run:
    uv run python -m evals.langsmith_eval

This is NOT part of CI. Run it manually after changing:
    - retrieval
    - embeddings
    - chunking
    - reranking
    - prompts
    - CRAG grading
    - AWS Docs fallback
"""

import os
import re

from app.config import settings

# ---------------------------------------------------------------------------
# LangSmith configuration
# ---------------------------------------------------------------------------

if settings.LANGSMITH_API_KEY:
    os.environ.setdefault(
        "LANGSMITH_API_KEY",
        settings.LANGSMITH_API_KEY,
    )

os.environ.setdefault(
    "LANGSMITH_PROJECT",
    settings.LANGSMITH_PROJECT or "aws-knowledge-assistant",
)

# Support both newer and older LangSmith tracing variables.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

from langsmith import Client
from langsmith.evaluation import evaluate

from app.agents.graph import run_query
from app.services.llm import generate_answer
from evals.evaluation import QUESTIONS, check_keywords, check_sources


DATASET_NAME = "aws-knowledge-assistant-eval"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_or_create_dataset(client: Client):
    """
    Create the persistent LangSmith dataset if it does not already exist.
    Otherwise, reuse the existing dataset.
    """

    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "AWS case-study RAG evaluation questions with expected "
            "keywords and expected source documents."
        ),
    )

    for item in QUESTIONS:
        client.create_example(
            inputs={
                "question": item["question"],
            },
            outputs={
                "expected_keywords": item["expected_keywords"],
                "expected_sources": item["expected_sources"],
            },
            dataset_id=dataset.id,
        )

    return dataset


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def target(inputs: dict) -> dict:
    """
    Function executed by LangSmith for every dataset example.
    """

    question = inputs["question"]

    result = run_query(question)

    return {
        "answer": result.get("answer", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "grade": result.get("grade"),
        "steps": result.get("steps", []),
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def proportion_to_score(value: float) -> float:
    """
    Convert a 0.0-1.0 proportion into a 1-10 score.

    Examples:
        0.0 -> 1
        0.5 -> 5.5
        1.0 -> 10
    """

    value = max(0.0, min(1.0, float(value)))

    # Map 0-1 to 1-10.
    return round(1.0 + (value * 9.0), 1)


def extract_score(text: str) -> int | None:
    """
    Extract a 1-10 integer score from an LLM response.

    Handles responses such as:
        "9"
        "9/10"
        "Score: 8"
        "8 - Mostly supported"
    """

    if not text:
        return None

    match = re.search(r"\b(10|[1-9])\b", text.strip())

    if not match:
        return None

    score = int(match.group(1))

    return max(1, min(10, score))


def get_context(chunks: list) -> str:
    """
    Convert retrieved chunks into text for the evaluator LLM.
    """

    texts = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text", "")

        if text:
            texts.append(str(text))

    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Evaluator 1: Keyword Coverage
# ---------------------------------------------------------------------------

def keyword_coverage_evaluator(run, example) -> dict:
    """
    Measures how many expected keywords/concepts appear in the final answer.

    Original calculation:
        matched / expected

    Converted to:
        1-10 score
    """

    expected = example.outputs.get("expected_keywords", []) or []

    answer = ""

    if run.outputs:
        answer = run.outputs.get("answer", "") or ""

    matched = check_keywords(answer, expected)

    if expected:
        raw_score = len(matched) / len(expected)
    else:
        raw_score = 1.0

    score = proportion_to_score(raw_score)

    return {
        "key": "keyword_coverage",
        "score": score,
        "comment": (
            f"Score: {score}/10 | "
            f"Matched {len(matched)}/{len(expected)} expected keywords: "
            f"{matched}"
        ),
    }


# ---------------------------------------------------------------------------
# Evaluator 2: Source Coverage
# ---------------------------------------------------------------------------

def source_coverage_evaluator(run, example) -> dict:
    """
    Measures whether the expected source documents were retrieved.

    Converted to a 1-10 score.
    """

    expected = example.outputs.get("expected_sources", []) or []

    chunks = []

    if run.outputs:
        chunks = run.outputs.get("retrieved_chunks", []) or []

    matched = check_sources(chunks, expected)

    if expected:
        raw_score = len(matched) / len(expected)
    else:
        raw_score = 1.0

    score = proportion_to_score(raw_score)

    return {
        "key": "source_coverage",
        "score": score,
        "comment": (
            f"Score: {score}/10 | "
            f"Matched {len(matched)}/{len(expected)} expected sources: "
            f"{matched}"
        ),
    }


# ---------------------------------------------------------------------------
# Evaluator 3: Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """
You are evaluating the faithfulness of an answer produced by an AWS
case-study RAG assistant.

Your job is to determine whether the answer is supported by the
retrieved context.

IMPORTANT:
- Do NOT judge whether the answer is well-written.
- Do NOT judge whether the answer is useful.
- Only judge whether the claims in the answer are supported by the
  retrieved context.
- If the answer contains claims that are not supported by the context,
  reduce the score.
- Do not assume information that is not explicitly supported by the
  retrieved context.

Scoring:

10 = Every important claim is clearly supported by the context.
9  = Almost completely supported; only extremely minor uncertainty.
8  = Mostly supported with a small amount of uncertainty.
7  = Generally supported but contains a minor unsupported claim.
6  = Some unsupported claims, but the main answer is supported.
5  = Mixed; roughly half of the answer is supported.
4  = Several important unsupported claims.
3  = Many unsupported claims.
2  = Most of the answer is unsupported.
1  = The answer is almost entirely unsupported by the context.

Retrieved context:
{context}

Answer:
{answer}

Return ONLY one integer from 1 to 10.
"""


def faithfulness_evaluator(run, example) -> dict:
    """
    LLM-as-judge faithfulness evaluator.

    Uses the application's existing generate_answer() function,
    which means it uses the same configured LLM gateway/provider
    rather than requiring a separate Anthropic API key.
    """

    chunks = []

    if run.outputs:
        chunks = run.outputs.get("retrieved_chunks", []) or []

    answer = ""

    if run.outputs:
        answer = run.outputs.get("answer", "") or ""

    if not chunks:
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "No retrieved context available for faithfulness evaluation.",
        }

    if not answer.strip():
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "No answer available for faithfulness evaluation.",
        }

    context = get_context(chunks)

    if not context.strip():
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "Retrieved chunks contained no usable text.",
        }

    prompt = _FAITHFULNESS_PROMPT.format(
        context=context,
        answer=answer,
    )

    try:
        raw = generate_answer(prompt).strip()

        score = extract_score(raw)

        if score is None:
            return {
                "key": "faithfulness",
                "score": None,
                "comment": (
                    f"Judge returned an invalid score: {raw[:200]}"
                ),
            }

        return {
            "key": "faithfulness",
            "score": score,
            "comment": f"Faithfulness score: {score}/10",
        }

    except Exception as exc:
        return {
            "key": "faithfulness",
            "score": None,
            "comment": f"Judge call failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Evaluator 4: Answer Correctness
# ---------------------------------------------------------------------------

_CORRECTNESS_PROMPT = """
You are evaluating the quality and correctness of an answer produced by
an AWS case-study RAG assistant.

Evaluate the answer using:
1. The user's question.
2. The retrieved context.
3. The expected keywords/concepts.

Consider:
- Does the answer directly answer the question?
- Is the information factually consistent with the retrieved context?
- Does it include the important expected information?
- Does it avoid making major incorrect claims?
- Is it sufficiently complete for the question?

Scoring:

10 = Excellent, correct, complete, and directly answers the question.
9  = Correct with only very minor omissions.
8  = Correct and useful with some minor missing information.
7  = Mostly correct but has noticeable omissions.
6  = Generally correct but incomplete or somewhat unclear.
5  = Mixed quality; partially correct.
4  = Contains significant incorrect or missing information.
3  = Mostly incorrect or poorly answers the question.
2  = Very poor answer with major factual problems.
1  = Completely incorrect or does not answer the question.

Question:
{question}

Expected keywords/concepts:
{expected_keywords}

Retrieved context:
{context}

Answer:
{answer}

Return ONLY one integer from 1 to 10.
"""


def answer_correctness_evaluator(run, example) -> dict:
    """
    LLM-as-judge evaluator for overall answer correctness.
    """

    question = ""

    if example.inputs:
        question = example.inputs.get("question", "") or ""

    expected_keywords = []

    if example.outputs:
        expected_keywords = (
            example.outputs.get("expected_keywords", []) or []
        )

    chunks = []

    if run.outputs:
        chunks = run.outputs.get("retrieved_chunks", []) or []

    answer = ""

    if run.outputs:
        answer = run.outputs.get("answer", "") or ""

    if not answer.strip():
        return {
            "key": "answer_correctness",
            "score": 1,
            "comment": "No answer was produced.",
        }

    context = get_context(chunks)

    prompt = _CORRECTNESS_PROMPT.format(
        question=question,
        expected_keywords=", ".join(map(str, expected_keywords)),
        context=context if context else "No retrieved context.",
        answer=answer,
    )

    try:
        raw = generate_answer(prompt).strip()

        score = extract_score(raw)

        if score is None:
            return {
                "key": "answer_correctness",
                "score": None,
                "comment": (
                    f"Judge returned an invalid score: {raw[:200]}"
                ),
            }

        return {
            "key": "answer_correctness",
            "score": score,
            "comment": f"Answer correctness score: {score}/10",
        }

    except Exception as exc:
        return {
            "key": "answer_correctness",
            "score": None,
            "comment": f"Judge call failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Create/reuse the dataset and run the LangSmith experiment.
    """

    if not settings.LANGSMITH_API_KEY:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. "
            "Add your LangSmith API key to .env."
        )

    print("=" * 70)
    print("AWS Knowledge Assistant - LangSmith Evaluation")
    print("=" * 70)

    print(f"Dataset : {DATASET_NAME}")
    print(
        f"Project : "
        f"{settings.LANGSMITH_PROJECT or 'aws-knowledge-assistant'}"
    )

    client = Client()

    dataset = get_or_create_dataset(client)

    print(f"Dataset ID: {dataset.id}")
    print()
    print("Running evaluation...")
    print()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            keyword_coverage_evaluator,
            source_coverage_evaluator,
            faithfulness_evaluator,
            answer_correctness_evaluator,
        ],
        experiment_prefix="aws-knowledge-assistant",
        client=client,
    )

    print()
    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)
    print()
    print("View the experiment in LangSmith:")
    print(
        f"https://smith.langchain.com/"
    )
    print()
    print(results)


if __name__ == "__main__":
    main()