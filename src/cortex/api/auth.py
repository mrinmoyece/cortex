"""
Cortex API authentication.

JWT-based auth with two token types:
  - Bearer tokens (short-lived, 1 hour) — for interactive API use
  - API keys (long-lived) — for service-to-service / agent clients

Both are validated in get_current_user() which is a FastAPI dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

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
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=ALGORITHM)


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
        )
        return TokenPayload(**payload)
    except JWTError as exc:
        if "expired" in str(exc).lower():
            raise HTTPException(status_code=401, detail="Token expired") from exc
        raise HTTPException(status_code=401, detail="Invalid token") from exc
