"""
API authentication.

A single shared API key, passed as the `X-API-Key` header, gates
every request to the app. This is intentionally simple (no user
accounts, no JWT) — swap for OAuth2/JWT if you need per-user
identity later; the dependency injection point in main.py stays
the same either way.
"""

from fastapi import Header, HTTPException, status

from app.config import settings


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """
    FastAPI dependency. Raises 401 if the request's X-API-Key header
    doesn't match the configured key.

    If API_KEY is left unset in the environment, auth is disabled —
    this is intentional for local dev, but means you MUST set API_KEY
    before deploying anywhere reachable from outside your machine.
    """

    if not settings.API_KEY:
        return

    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
