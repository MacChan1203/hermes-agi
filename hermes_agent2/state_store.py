from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "state2.db"
SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    title TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    metadata_json TEXT,
    message_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    finish_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionDB:
    """SQLite + FTS5 の軽量セッション保存。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL)
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        cur.executescript(FTS_SQL)
        self._conn.commit()

    def create_session(self, session_id: str, *, source: str = "cli", model: str | None = None, title: str | None = None, user_id: str | None = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, source, user_id, model, title, started_at, metadata_json, message_count) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT message_count FROM sessions WHERE id = ?), 0))",
                (session_id, source, user_id, model, title, time.time(), json.dumps(metadata or {}, ensure_ascii=False), session_id),
            )
            self._conn.commit()

    def append_message(self, session_id: str, role: str, content: str, *, tool_name: str | None = None, finish_reason: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, tool_name, timestamp, finish_reason) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tool_name, time.time(), finish_reason),
            )
            self._conn.execute("UPDATE sessions SET message_count = message_count + 1 WHERE id = ?", (session_id,))
            self._conn.commit()

    def end_session(self, session_id: str, reason: str = "completed") -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?", (time.time(), reason, session_id))
            self._conn.commit()

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT m.session_id, m.role, m.content, m.timestamp FROM messages_fts f JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? ORDER BY m.timestamp DESC LIMIT ?",
            (query, limit),
        )
        return [dict(row) for row in cur.fetchall()]
