"""Real user store: SQLite + bcrypt password hashing. Any user can sign up
(no invite/approval flow) - self-service accounts, same as the request.

Uses the `bcrypt` package directly rather than passlib's CryptContext -
passlib (last released 2020, effectively unmaintained) misdetects bcrypt
>=4.1's API and raises a spurious "password too long" error even for short
passwords; calling bcrypt.hashpw/checkpw directly sidesteps that entirely.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import bcrypt

from utils.core.paths import METADATA_DIR

DB_PATH = METADATA_DIR / "users.db"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # token_version: bumped on every password change. Password-reset
        # JWTs embed the version they were issued against (see
        # utils/auth/tokens.issue_reset_token) so a reset link is single-use -
        # using it bumps the version, which invalidates that link and any
        # other outstanding ones - without needing a separate token table.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "token_version" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
        # auth_provider: "local" (self-service password, the original/default)
        # or "auth0" (utils/auth/auth0.py). Existing rows default to "local" -
        # they were all created before Auth0 existed. Lets the Auth0 callback
        # refuse to silently take over a username someone already registered
        # with a password (see apps/api/routers/auth.py auth0_callback) -
        # an account-preregistration hijack this column exists specifically
        # to close, not just bookkeeping.
        if "auth_provider" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'local'")
        conn.commit()


_init_db(DB_PATH)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_user(username: str, password: str, auth_provider: str = "local") -> dict:
    """Raises ValueError if the username is already taken. password is a
    real bcrypt-hashed credential for auth_provider="local"; for "auth0" the
    caller (utils/auth/auth0.py's callback route) passes a securely random
    string instead - that account can only ever be reached by signing back
    in through Auth0, since nobody knows (or could reasonably guess) it."""
    with _connect() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise ValueError(f"username '{username}' is already taken")
        password_hash = _hash_password(password)
        created_at = time.time()
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at, auth_provider) VALUES (?, ?, ?, ?)",
            (username, password_hash, created_at, auth_provider),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username, "created_at": created_at, "auth_provider": auth_provider}


def verify_user(username: str, password: str) -> dict | None:
    """Returns the user record if username/password match, else None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def get_user(username: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, created_at, token_version, auth_provider FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def set_password(username: str, new_password: str) -> bool:
    """Overwrites the stored password hash and bumps token_version, which
    invalidates any outstanding reset link for this user (including the
    one just used - see the reset-password route). Returns False if the
    user doesn't exist (caller decides whether to surface that or stay
    silent - the forgot-password route stays silent to avoid leaking which
    usernames are registered)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ?, token_version = token_version + 1 WHERE username = ?",
            (_hash_password(new_password), username),
        )
        conn.commit()
        return cur.rowcount > 0
