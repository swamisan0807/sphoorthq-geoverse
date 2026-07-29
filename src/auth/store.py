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

from src.core.paths import METADATA_DIR

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


def create_user(username: str, password: str) -> dict:
    """Raises ValueError if the username is already taken."""
    with _connect() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise ValueError(f"username '{username}' is already taken")
        password_hash = _hash_password(password)
        created_at = time.time()
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, created_at),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username, "created_at": created_at}


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
            "SELECT id, username, created_at FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None
