"""
Planner node.

Routes:
- conversation -> direct responder
- rag -> retrieval + responder

Classification is two-tiered:
1. A fast, free exact-match against a small phrase list handles the
   overwhelming majority of conversational openers ("hi", "thanks",
   "who are you") with zero latency and zero LLM cost.
2. Anything short that doesn't match falls through to a cheap LLM
   classification call instead of defaulting straight to "rag" --
   this catches greetings/small talk the phrase list doesn't know
   about ("yo", "sup", typos, other languages) without sending every
   such message through the full retrieval pipeline.
   Longer messages skip the LLM call and go straight to "rag": a
   long message is overwhelmingly likely to be a real question, so
   spending an extra LLM round trip on it just adds latency.
"""

import logfire

from app.services.llm import generate_answer


CONVERSATION_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}

CONVERSATION_PREFIXES = (
    "who are you",
    "what can you do",
    "how can you help",
    "what do you do",
)

# Above this many words, skip the LLM classification fallback and
# route straight to RAG -- short-circuits the common case cheaply.
_SHORT_MESSAGE_WORD_LIMIT = 6

_INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for an \
AWS case-study Q&A assistant.

Decide whether the user's message is:
- "conversation" -- a greeting, small talk, thanks, farewell, or a \
question about the assistant itself (who/what it is, what it can do)
- "rag" -- an actual question that needs the AWS knowledge base

Message: "{message}"

Answer with ONLY one word: conversation or rag"""


def _phrase_match(normalized: str) -> bool:
    if normalized in CONVERSATION_PHRASES:
        return True

    return any(
        normalized.startswith(prefix)
        for prefix in CONVERSATION_PREFIXES
    )


def _classify_with_llm(question: str) -> bool:
    """
    Ask a cheap LLM call whether a short message is conversational.
    Defaults to "rag" (the safer failure mode -- a false "conversation"
    would silently skip retrieval) if the call fails or returns
    anything ambiguous.
    """

    try:
        raw = generate_answer(
            _INTENT_CLASSIFICATION_PROMPT.format(message=question)
        )
    except Exception as exc:
        logfire.error(
            "Intent classification LLM call failed, defaulting to rag",
            error=str(exc),
        )
        return False

    return raw.strip().lower().startswith("conversation")


def is_conversation(question: str) -> bool:
    normalized = " ".join(question.lower().split())

    if _phrase_match(normalized):
        return True

    if len(normalized.split()) <= _SHORT_MESSAGE_WORD_LIMIT:
        return _classify_with_llm(question)

    return False


def plan_node(state: dict) -> dict:

    with logfire.span("planner"):

        question = state["question"].strip()

        if is_conversation(question):
            intent = "conversation"
            search_query = ""
        else:
            intent = "rag"
            search_query = question

        logfire.info(
            "Planning completed",
            intent=intent,
            question=question,
        )

        state["intent"] = intent
        state["search_query"] = search_query

        state["steps"] = state.get("steps", []) + [
            {
                "node": "planner",
                "output": intent,
            }
        ]

        return state
