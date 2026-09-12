
"""
Planner node.

Routes:
- conversation -> direct responder
- mcp -> live AWS Documentation MCP -> responder
- rag -> local AWS case-study/blog RAG pipeline

The planner uses:
1. Fast deterministic rules for obvious procedural/documentation
   AWS questions.
2. Conversation detection for greetings/small talk.
3. RAG as the default for AWS knowledge-base questions.
"""

import logfire

from app.services.llm import generate_answer


# -------------------------------------------------------------------
# Conversation detection
# -------------------------------------------------------------------

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

_SHORT_MESSAGE_WORD_LIMIT = 6


# -------------------------------------------------------------------
# MCP routing
# -------------------------------------------------------------------

# Questions that explicitly ask for current AWS documentation,
# configuration, creation, setup, deployment, CLI/API instructions,
# policies, or operational procedures should go directly to MCP.
#
# This is intentionally specific so that:
#
# "How does Netflix use S3?"
# still goes to RAG
#
# while:
#
# "How do I create an S3 bucket?"
# goes to MCP.
MCP_PHRASES = (
    "how do i create",
    "how do i configure",
    "how do i setup",
    "how do i set up",
    "how do i enable",
    "how do i disable",
    "how do i deploy",
    "how do i install",
    "how to create",
    "how to configure",
    "how to setup",
    "how to set up",
    "how to enable",
    "how to disable",
    "how to deploy",
    "how to install",
    "aws cli",
    "aws documentation",
    "aws docs",
    "aws api",
    "aws sdk",
)

MCP_OPERATION_WORDS = (
    "create",
    "configure",
    "setup",
    "set up",
    "enable",
    "disable",
    "deploy",
    "install",
    "troubleshoot",
)

AWS_SERVICE_NAMES = (
    "s3",
    "ec2",
    "lambda",
    "dynamodb",
    "rds",
    "cloudfront",
    "vpc",
    "iam",
    "ecs",
    "eks",
    "sns",
    "sqs",
    "cloudwatch",
    "route 53",
    "route53",
)


def _normalize(question: str) -> str:
    return " ".join(question.lower().split())


def _phrase_match(normalized: str) -> bool:
    if normalized in CONVERSATION_PHRASES:
        return True

    return any(
        normalized.startswith(prefix)
        for prefix in CONVERSATION_PREFIXES
    )


def is_mcp_question(question: str) -> bool:
    """
    Detect procedural/documentation-oriented AWS questions.

    We deliberately avoid treating every "how" question as MCP,
    because case-study questions such as:

        "How does Netflix use S3?"

    should still use RAG.
    """
    normalized = _normalize(question)

    # Explicit documentation / API / CLI requests.
    if any(phrase in normalized for phrase in MCP_PHRASES):
        return True

    # "how do I..." / "how to..." combined with an AWS service.
    procedural_prefixes = (
        "how do i",
        "how to",
    )

    if any(normalized.startswith(prefix) for prefix in procedural_prefixes):
        return any(
            service in normalized
            for service in AWS_SERVICE_NAMES
        )

    # Operational questions containing an AWS service.
    has_operation = any(
        word in normalized
        for word in MCP_OPERATION_WORDS
    )

    has_aws_service = any(
        service in normalized
        for service in AWS_SERVICE_NAMES
    )

    if has_operation and has_aws_service:
        return True

    return False


# -------------------------------------------------------------------
# LLM conversation classifier
# -------------------------------------------------------------------

_INTENT_CLASSIFICATION_PROMPT = """
You are an intent classifier for an AWS knowledge assistant.

Classify the user's message as exactly ONE of:

- "conversation"
  Greetings, small talk, thanks, farewell, or questions about
  the assistant itself.

- "mcp"
  AWS documentation, AWS service configuration, resource creation,
  deployment, troubleshooting, AWS CLI/API/SDK instructions,
  IAM policies, or procedural "how do I..." / "how to..." questions.

- "rag"
  Questions about AWS case studies, AWS customer stories,
  AWS technical blogs, architectures described in those sources,
  AWS service usage by companies, comparisons of companies,
  and other knowledge-base questions.

Examples:

"Hi"
-> conversation

"What can you do?"
-> conversation

"How do I create an S3 bucket?"
-> mcp

"How do I enable S3 versioning?"
-> mcp

"What is the S3 bucket policy syntax?"
-> mcp

"What AWS services does Netflix use?"
-> rag

"How does Netflix use S3?"
-> rag

"Compare Netflix and Slack's AWS architecture."
-> rag

Message:
"{message}"

Answer with ONLY one word:
conversation
mcp
or
rag
"""


def _classify_with_llm(question: str) -> str:
    """
    Classify short ambiguous questions.

    Defaults to RAG because it is safer than accidentally skipping
    the knowledge-base pipeline.
    """
    try:
        raw = generate_answer(
            _INTENT_CLASSIFICATION_PROMPT.format(
                message=question
            )
        )

        result = raw.strip().lower()

        if result.startswith("conversation"):
            return "conversation"

        if result.startswith("mcp"):
            return "mcp"

        return "rag"

    except Exception as exc:
        logfire.error(
            "Intent classification LLM call failed, defaulting to rag",
            error=str(exc),
        )
        return "rag"


def plan_node(state: dict) -> dict:
    with logfire.span("planner"):

        question = state["question"].strip()
        normalized = _normalize(question)

        # ---------------------------------------------------------
        # 1. Conversation
        # ---------------------------------------------------------

        if _phrase_match(normalized):
            intent = "conversation"

        # ---------------------------------------------------------
        # 2. Deterministic MCP routing
        # ---------------------------------------------------------

        elif is_mcp_question(question):
            intent = "mcp"

        # ---------------------------------------------------------
        # 3. Short ambiguous messages
        # ---------------------------------------------------------

        elif len(normalized.split()) <= _SHORT_MESSAGE_WORD_LIMIT:
            intent = _classify_with_llm(question)

        # ---------------------------------------------------------
        # 4. Normal AWS knowledge question
        # ---------------------------------------------------------

        else:
            intent = "rag"

        if intent == "conversation":
            search_query = ""

        else:
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


