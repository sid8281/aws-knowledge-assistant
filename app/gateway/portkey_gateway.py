"""
LLM Gateway.

All application LLM calls go through Portkey.

Portkey provides:
- Primary LLM provider
- Automatic fallback provider
- Centralized LLM routing

Logfire provides:
- LLM call tracing
- Request latency
- Success/failure
- Response size
"""

import time

import logfire
from portkey_ai import Portkey

from app.config import settings


# -------------------------------------------------------------------
# Portkey fallback configuration
# -------------------------------------------------------------------

_FALLBACK_CONFIG = {
    "strategy": {
        "mode": "fallback",
    },
    "targets": [
        {
            "virtual_key": settings.PORTKEY_VIRTUAL_KEY_PRIMARY,
            "override_params": {
                "model": settings.LLM_MODEL_PRIMARY,
            },
        },
        {
            "virtual_key": settings.PORTKEY_VIRTUAL_KEY_FALLBACK,
            "override_params": {
                "model": settings.LLM_MODEL_FALLBACK,
            },
        },
    ],
}


# Prefer saved Portkey config when available.
# Otherwise use inline fallback configuration.

_config = (
    settings.PORTKEY_CONFIG_SLUG
    or _FALLBACK_CONFIG
)


_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=_config,
)


# -------------------------------------------------------------------
# LLM generation
# -------------------------------------------------------------------

def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    """
    Send a chat completion request through Portkey
    and return the generated text.
    """

    start_time = time.perf_counter()

    with logfire.span(
        "llm.generate",
    ):

        logfire.info(
            "LLM request started",
            model=settings.LLM_MODEL_PRIMARY,
            temperature=temperature,
            prompt_length=len(user_prompt),
        )

        try:

            response = _client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
            )

            result = response.choices[0].message.content or ""

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logfire.info(
                "LLM request completed",
                model=settings.LLM_MODEL_PRIMARY,
                response_length=len(result),
                latency_ms=round(elapsed_ms, 2),
            )

            return result

        except Exception as exc:

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logfire.error(
                "LLM request failed",
                error=str(exc),
                latency_ms=round(elapsed_ms, 2),
            )

            raise