
"""
Rewrites follow-up questions into standalone search queries.

Also used as the retry step in the CRAG loop (see grader.py): when
re-entered after the grader marks the previous retrieval "ambiguous",
it rewrites the query with an explicit instruction to broaden or
rephrase the search rather than just resolving conversational
references, since the earlier standalone query already produced
weak results.
"""

from app.services.llm import generate_answer


_STANDARD_INSTRUCTIONS = """Use the conversation history to resolve:
- pronouns such as "it", "they", "that"
- follow-ups such as "what about Slack?"
- omitted entities
- references to previous answers"""

_RETRY_INSTRUCTIONS = """The previous search query below did not \
retrieve clearly relevant results:
"{previous_query}"

Rewrite the question as a broader or differently-phrased standalone \
search query -- try different terminology, a related AWS service \
name, or a more general phrasing of the same underlying question."""


def rewrite_query(state: dict) -> dict:

    question = state["question"]
    messages = state.get("messages", [])
    is_retry = state.get("grade") == "ambiguous"

    # No previous conversation and not a CRAG retry -- nothing to
    # resolve, use the question as-is.
    if not messages and not is_retry:
        return {
            "search_query": question,
            "steps": state.get("steps", []) + [
                {
                    "node": "query_rewriter",
                    "output": "No conversation history",
                }
            ],
        }

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages[-6:]
    )

    instructions = (
        _RETRY_INSTRUCTIONS.format(
            previous_query=state.get("search_query", question)
        )
        if is_retry
        else _STANDARD_INSTRUCTIONS
    )

    prompt = f"""
You are a search query rewriting component for an AWS case-study RAG system.

Rewrite the user's latest question into a standalone search query.

{instructions}

Do NOT answer the question.

Return ONLY the rewritten search query.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone search query:
"""

    rewritten = generate_answer(prompt).strip()

    # Safety fallback
    if not rewritten:
        rewritten = question

    return {
        "search_query": rewritten,
        "steps": state.get("steps", []) + [
            {
                "node": "query_rewriter",
                "output": rewritten,
            }
        ],
    }