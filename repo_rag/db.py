import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
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


def create_chat(title: str) -> dict:
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
    return {"id": cur.lastrowid, "title": title, "created_at": now, "updated_at": now}


def list_chats() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id: int) -> dict | None:
    """One chat with its messages, sources/metrics JSON already decoded."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
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


def delete_chat(chat_id: int) -> bool:
    with _db() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
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
