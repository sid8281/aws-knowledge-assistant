
import json
import time
from pathlib import Path

from app.agents.graph import run_query


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "question": "What AWS services are mentioned in the Netflix and Slack case studies?",
        "expected_keywords": [
            "Amazon EC2",
            "Amazon S3",
            "Amazon CloudFront",
            "Amazon RDS",
        ],
        "expected_sources": ["Netflix", "Slack"],
    },
    {
        "question": "Which AWS service is used for content delivery in the Netflix case study?",
        "expected_keywords": ["Amazon CloudFront"],
        "expected_sources": ["Netflix"],
    },
    {
        "question": "Which AWS service is used for storage in the Netflix and Slack case studies?",
        "expected_keywords": ["Amazon S3"],
        "expected_sources": ["Netflix", "Slack"],
    },
    {
        "question": "Which AWS service does Slack use for its database?",
        "expected_keywords": ["Amazon RDS"],
        "expected_sources": ["Slack"],
    },
    {
        "question": "Which AWS services are common between Netflix and Slack?",
        "expected_keywords": ["Amazon EC2", "Amazon S3"],
        "expected_sources": ["Netflix", "Slack"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def check_keywords(answer: str, expected_keywords: list[str]) -> list[str]:
    answer_normalized = normalize(answer)

    return [
        keyword
        for keyword in expected_keywords
        if normalize(keyword) in answer_normalized
    ]


def check_sources(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> list[str]:
    retrieved_text = normalize(
        "\n".join(
            chunk.get("text", "")
            for chunk in retrieved_chunks
        )
    )

    return [
        source
        for source in expected_sources
        if normalize(source) in retrieved_text
    ]


# ---------------------------------------------------------------------------
# Evaluate one question
# ---------------------------------------------------------------------------

def evaluate_question(item: dict) -> dict:
    question = item["question"]

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    start_time = time.perf_counter()

    try:
        result = run_query(question)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        print(f"\nRAG ERROR: {exc}")

        return {
            "question": question,
            "status": "rag_error",
            "latency_seconds": round(elapsed, 3),
            "error": str(exc),
        }

    elapsed = time.perf_counter() - start_time

    answer = result.get("answer", "")
    retrieved_chunks = result.get("retrieved_chunks", [])

    matched_keywords = check_keywords(
        answer,
        item["expected_keywords"],
    )

    matched_sources = check_sources(
        retrieved_chunks,
        item["expected_sources"],
    )

    keyword_score = (
        len(matched_keywords) / len(item["expected_keywords"])
        if item["expected_keywords"]
        else 0
    )

    source_score = (
        len(matched_sources) / len(item["expected_sources"])
        if item["expected_sources"]
        else 0
    )

    # A question passes when the expected answer information
    # and expected source documents were both retrieved.
    passed = keyword_score == 1.0 and source_score == 1.0

    print("\nANSWER")
    print("-" * 80)
    print(answer)

    print("\nRESULT")
    print("-" * 80)
    print(f"Retrieved chunks : {len(retrieved_chunks)}")
    print(f"Matched keywords : {len(matched_keywords)}/{len(item['expected_keywords'])}")
    print(f"Matched sources  : {len(matched_sources)}/{len(item['expected_sources'])}")
    print(f"Keyword score    : {keyword_score:.2f}")
    print(f"Source score     : {source_score:.2f}")
    print(f"Latency          : {elapsed:.3f}s")
    print(f"Status           : {'PASS' if passed else 'FAIL'}")

    return {
        "question": question,
        "status": "passed" if passed else "failed",
        "answer": answer,
        "retrieved_chunks": len(retrieved_chunks),
        "matched_keywords": matched_keywords,
        "expected_keywords": item["expected_keywords"],
        "matched_sources": matched_sources,
        "expected_sources": item["expected_sources"],
        "keyword_score": round(keyword_score, 3),
        "source_score": round(source_score, 3),
        "latency_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("AWS CASE STUDIES RAG EVALUATION")
    print("=" * 80)

    print(f"Evaluation questions: {len(QUESTIONS)}")

    results = []

    for item in QUESTIONS:
        result = evaluate_question(item)
        results.append(result)

    successful = [
        result
        for result in results
        if result["status"] in ("passed", "failed")
    ]

    passed = [
        result
        for result in successful
        if result["status"] == "passed"
    ]

    rag_errors = [
        result
        for result in results
        if result["status"] == "rag_error"
    ]

    if successful:
        accuracy = len(passed) / len(successful)

        avg_latency = sum(
            result["latency_seconds"]
            for result in successful
        ) / len(successful)

        avg_keyword_score = sum(
            result["keyword_score"]
            for result in successful
        ) / len(successful)

        avg_source_score = sum(
            result["source_score"]
            for result in successful
        ) / len(successful)
    else:
        accuracy = 0
        avg_latency = 0
        avg_keyword_score = 0
        avg_source_score = 0

    summary = {
        "total_questions": len(QUESTIONS),
        "completed": len(successful),
        "passed": len(passed),
        "rag_errors": len(rag_errors),
        "accuracy": round(accuracy, 3),
        "average_keyword_score": round(avg_keyword_score, 3),
        "average_source_score": round(avg_source_score, 3),
        "average_latency_seconds": round(avg_latency, 3),
        "results": results,
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    output_path = Path(__file__).parent / "evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total questions       : {summary['total_questions']}")
    print(f"Completed             : {summary['completed']}")
    print(f"Passed                : {summary['passed']}")
    print(f"RAG errors            : {summary['rag_errors']}")
    print(f"Accuracy              : {summary['accuracy']:.2%}")
    print(f"Avg keyword score     : {summary['average_keyword_score']:.2%}")
    print(f"Avg source score      : {summary['average_source_score']:.2%}")
    print(f"Avg latency           : {summary['average_latency_seconds']:.3f}s")

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
