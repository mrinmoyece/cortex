"""
Cortex API authentication.

JWT-based auth with two token types:
  - Bearer tokens (short-lived, 1 hour) — for interactive API use
  - API keys (long-lived) — for service-to-service / agent clients

Both are validated in get_current_user() which is a FastAPI dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError
from pydantic import BaseModel, ValidationError

from cortex.config import settings

#: `datetime.UTC` is 3.11+; this package supports 3.10.
UTC = timezone.utc

_bearer = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class TokenPayload(BaseModel):
    sub: str  # user_id
    tenant: str = "default"
    scopes: list[str] = []
    exp: int | None = None


def create_access_token(
    user_id: str, tenant: str = "default", scopes: list[str] | None = None
) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "tenant": tenant,
        "scopes": scopes or [],
        "exp": expire,
    }
    token: str = jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=ALGORITHM)
    return token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenPayload:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        return TokenPayload(**payload)
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except InvalidTokenError as exc:
        # Substring-matching the exception message for "expired" was how
        # expiry used to be detected; PyJWT raises a distinct type, so the
        # two failure modes are now told apart by the library rather than
        # by prose.
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    except ValidationError as exc:
        # A validly-signed token with the wrong shape - no `sub`, a
        # non-list `scopes` - is a rejected credential, not a server fault.
        # It used to escape as a 500, which both leaked a stack trace and
        # made an authentication failure look like an outage.
        raise HTTPException(status_code=401, detail="Malformed token claims") from exc
