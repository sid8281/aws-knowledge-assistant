"""
Thin wrapper around NeMo Guardrails.

Two functions are exposed:
- check_input(text)  -> (is_allowed: bool, message: str or None)
- check_output(text) -> (is_allowed: bool, message: str or None)

main.py calls check_input() before the query reaches the RAG pipeline,
and check_output() on the generated answer before returning it.

Logfire is used to trace:
- input guardrail checks
- output guardrail checks
- allow/block decisions
- processing time
"""

import logfire

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationRailsOptions,
)

from app.config import settings


# -------------------------------------------------------------------
# NeMo Guardrails setup
# -------------------------------------------------------------------

_config = RailsConfig.from_path(
    settings.GUARDRAILS_CONFIG_DIR
)

_rails = LLMRails(_config)


# -------------------------------------------------------------------
# Refusal message
#
# This MUST match the literal text of `define bot refuse to respond`
# in app/guardrails/config/rails.co exactly -- per NeMo's documented
# behavior, a blocked message returns this predefined string verbatim
# (https://docs.nvidia.com/nemo/guardrails/latest/user-guides/advanced/generation-options.html#input-rails-only).
# tests/guardrails/test_guard.py has a drift check between the two.
# -------------------------------------------------------------------

REFUSAL_MESSAGE = (
    "I can't help with that. I can only answer questions "
    "about AWS and the AWS knowledge base."
)


# -------------------------------------------------------------------
# Guardrail options
# -------------------------------------------------------------------

# Only the input rail runs.
# Dialog generation is disabled because we only want a yes/no decision.

_INPUT_CHECK_OPTIONS = GenerationOptions(
    rails=GenerationRailsOptions(
        input=True,
        output=False,
        retrieval=False,
        dialog=False,
    )
)


# Only the output rail runs.
# Dialog generation is disabled for the same reason.

_OUTPUT_CHECK_OPTIONS = GenerationOptions(
    rails=GenerationRailsOptions(
        input=False,
        output=True,
        retrieval=False,
        dialog=False,
    )
)


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------

def _extract_content(result) -> str:
    """
    Pull text out of a GenerationResponse.

    Returns an empty string if the rail didn't generate
    a response.
    """

    if not result or not result.response:
        return ""

    return result.response[0].get(
        "content",
        "",
    ) or ""


def _was_blocked(content: str) -> bool:
    """
    Decide whether a rail check blocked the message.

    NeMo returns the fixed `bot refuse to respond` string verbatim
    when a rail blocks -- and only then. Comparing against that known
    constant (instead of comparing against the variable input/output
    text) means alterations a rail makes to allowed content (e.g.
    whitespace normalization, PII masking) are never mistaken for a
    block, which the previous echo-comparison approach got wrong.
    """

    return content.strip() == REFUSAL_MESSAGE


# -------------------------------------------------------------------
# Input Guardrail
# -------------------------------------------------------------------

def check_input(
    user_text: str,
) -> tuple[bool, str | None]:
    """
    Check whether the user input is allowed.

    Returns:
        (True, None)   -> allowed
        (False, msg)   -> blocked
    """

    with logfire.span(
        "guard.input",
    ):

        logfire.info(
            "Input guardrail started",
            text_length=len(user_text),
        )

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
            options=_INPUT_CHECK_OPTIONS,
        )

        content = _extract_content(result).strip()

        if _was_blocked(content):

            logfire.info(
                "Input guardrail blocked request",
                decision="block",
            )

            return False, content

        logfire.info(
            "Input guardrail allowed request",
            decision="allow",
        )

        return True, None


# -------------------------------------------------------------------
# Output Guardrail
# -------------------------------------------------------------------

def check_output(
    bot_text: str,
) -> tuple[bool, str]:
    """
    Check whether the generated bot response is allowed.

    Returns:
        (True, bot_text) -> allowed
        (False, msg)     -> blocked
    """

    with logfire.span(
        "guard.output",
    ):

        logfire.info(
            "Output guardrail started",
            text_length=len(bot_text),
        )

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "(no user message — "
                        "checking bot output only)"
                    ),
                },
                {
                    "role": "assistant",
                    "content": bot_text,
                },
            ],
            options=_OUTPUT_CHECK_OPTIONS,
        )

        content = _extract_content(result).strip()

        if _was_blocked(content):

            logfire.info(
                "Output guardrail blocked response",
                decision="block",
            )

            return False, content

        logfire.info(
            "Output guardrail allowed response",
            decision="allow",
        )

        return True, bot_text