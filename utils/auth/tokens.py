"""JWT session tokens. Secret is generated once and persisted to disk (not
committed to git - see .gitignore) so sessions survive a backend restart;
set JWT_SECRET in the environment to override for a real deployment."""

import os
import secrets
import time

import jwt

from utils.core.paths import METADATA_DIR

SECRET_PATH = METADATA_DIR / ".jwt_secret"
ALGORITHM = "HS256"
SESSION_TTL_SECONDS = 24 * 3600


def _load_or_create_secret() -> str:
    env_secret = os.environ.get("JWT_SECRET")
    if env_secret:
        return env_secret
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    SECRET_PATH.write_text(secret, encoding="utf-8")
    return secret


_SECRET = _load_or_create_secret()

RESET_TTL_SECONDS = 15 * 60


def issue_token(username: str) -> str:
    now = time.time()
    payload = {"sub": username, "iat": now, "exp": now + SESSION_TTL_SECONDS}
    return jwt.encode(payload, _SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> str | None:
    """Returns the username if the token is valid and unexpired, else None."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") is not None:
        # a reset token presented as a session token - reject it
        return None
    return payload.get("sub")


def issue_reset_token(username: str, token_version: int) -> str:
    """Short-lived, single-purpose token emailed as a password-reset link.
    Tagged with purpose='reset' so it can never be replayed as a session
    token (decode_token above rejects anything carrying a purpose claim),
    and carries the user's current token_version so the route can enforce
    single-use: utils.auth.store.set_password bumps token_version on every
    reset, so a version mismatch means this exact link was already used
    (or a newer one was issued since)."""
    now = time.time()
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + RESET_TTL_SECONDS,
        "purpose": "reset",
        "tv": token_version,
    }
    return jwt.encode(payload, _SECRET, algorithm=ALGORITHM)


def decode_reset_token(token: str) -> tuple[str, int] | None:
    """Returns (username, token_version) if this is a valid, unexpired
    reset token, else None. Caller still has to compare token_version
    against the user's current one in the DB to enforce single-use."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") != "reset":
        return None
    username = payload.get("sub")
    token_version = payload.get("tv")
    if username is None or token_version is None:
        return None
    return username, token_version
