"""JWT session tokens. Secret is generated once and persisted to disk (not
committed to git - see .gitignore) so sessions survive a backend restart;
set JWT_SECRET in the environment to override for a real deployment."""

import os
import secrets
import time

import jwt

from src.core.paths import METADATA_DIR

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
    return payload.get("sub")
