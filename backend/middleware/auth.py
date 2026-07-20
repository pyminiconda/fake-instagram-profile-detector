"""
middleware/auth.py — JWT creation and verification helpers.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from backend.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

bearer_scheme = HTTPBearer()

# ── In-memory reset token store (simple, works for academic project) ──
_reset_tokens: dict[str, dict] = {}   # token → {user_id, email, expires}


# ── Token creation ────────────────────────────────────────────────────

def create_access_token(user_id: str, username: str, is_admin: bool) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "is_admin": is_admin,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Token verification ────────────────────────────────────────────────

def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """FastAPI dependency — returns decoded token payload for authenticated routes."""
    payload = _decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type.")
    return payload


def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency — restricts route to admin users only."""
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return current_user


def decode_refresh_token(token: str) -> str:
    """Validate a refresh token and return the user_id (sub)."""
    payload = _decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type.")
    return payload["sub"]


# ── Password reset tokens ─────────────────────────────────────────────

def create_reset_token(user_id: str, email: str) -> str:
    """Generate a one-time password reset token (stored in memory)."""
    token = str(uuid.uuid4())
    _reset_tokens[token] = {
        "user_id": user_id,
        "email": email,
        "expires": datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    return token


def verify_reset_token(token: str) -> Optional[dict]:
    """Verify and consume a reset token. Returns payload or None."""
    entry = _reset_tokens.get(token)
    if not entry:
        return None
    if datetime.now(timezone.utc) > entry["expires"]:
        _reset_tokens.pop(token, None)
        return None
    _reset_tokens.pop(token, None)   # one-time use
    return entry
