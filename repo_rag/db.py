import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# DATA_DIR lets deployments (e.g. Render) point the store at a mounted disk;
# defaults to <project>/data locally.
DB_PATH = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data")) / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pw_hash    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL DEFAULT 0,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    sources    TEXT,
    metrics    TEXT,
    error      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with _LOCK:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.executescript(_SCHEMA)
        # pre-auth databases: chats existed without an owner column; existing
        # rows land on user_id 0 (legacy) until claimed by LEGACY_OWNER_NAME
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(chats)")}
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE chats ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0"
            )


def create_chat(title: str, user_id: int = 0) -> dict:
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
    return {"id": cur.lastrowid, "title": title, "created_at": now, "updated_at": now}


def list_chats(user_id: int | None = None) -> list[dict]:
    """Conversations, most recently active first; scoped to one user when given."""
    with _db() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats "
                "WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
                (user_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id: int) -> dict | None:
    """One chat with its messages, sources/metrics JSON already decoded."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT id, role, content, sources, metrics, error, created_at "
            "FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    chat = dict(row)
    chat["messages"] = [
        {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "sources": json.loads(m["sources"]) if m["sources"] else [],
            "metrics": json.loads(m["metrics"]) if m["metrics"] else None,
            "error": m["error"],
            "created_at": m["created_at"],
        }
        for m in messages
    ]
    return chat


def delete_chat(chat_id: int, user_id: int | None = None) -> bool:
    """Delete one chat; when user_id is given, only if that user owns it."""
    with _db() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        else:
            cur = conn.execute(
                "DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
            )
    return cur.rowcount > 0


def append_message(
    chat_id: int,
    role: str,
    content: str,
    *,
    sources: list | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> int:
    """Append one message and bump the chat's updated_at."""
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, metrics, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                chat_id,
                role,
                content,
                json.dumps(sources) if sources else None,
                json.dumps(metrics) if metrics else None,
                error,
                now,
            ),
        )
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    return cur.lastrowid


def chat_history(
    chat_id: int, max_messages: int = 8, max_chars: int = 3_000
) -> list[dict]:
    """Recent turns (oldest -> newest) for follow-up continuity in the prompt."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE chat_id = ? AND content != '' ORDER BY id DESC LIMIT ?",
            (chat_id, max_messages),
        ).fetchall()
    history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    while history and sum(len(m["content"]) for m in history) > max_chars:
        history.pop(0)
    return history


# --- users & sessions -----------------------------------------------------------


def create_user_row(username: str, pw_hash: str) -> dict:
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, pw_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, now),
        )
    return {"id": cur.lastrowid, "username": username, "created_at": now}


def get_user_by_username(username: str) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT id, username, pw_hash FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user(user_id: int) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def insert_session(token_hash: str, user_id: int, expires_at: str) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token_hash, user_id, _now(), expires_at),
        )


def get_session_user(token_hash: str) -> dict | None:
    """User for a live (unexpired) session token hash, else None."""
    with _db() as conn:
        conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (_now(),)
        )  # cheap opportunistic purge of stale rows
        row = conn.execute(
            "SELECT u.id, u.username FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at >= ?",
            (token_hash, _now()),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token_hash: str) -> bool:
    with _db() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    return cur.rowcount > 0


def claim_legacy_chats(new_user_id: int) -> int:
    """Reassign the pre-auth chats (user_id 0) to a real user; returns the count."""
    with _db() as conn:
        cur = conn.execute(
            "UPDATE chats SET user_id = ? WHERE user_id = 0", (new_user_id,)
        )
    return cur.rowcount
