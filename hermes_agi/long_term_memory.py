"""長期記憶エンジン。セッションをまたいで知識・戦略・失敗パターンを蓄積する。"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LTM_PATH = Path.home() / ".hermes" / "long_term_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    session_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_hash TEXT NOT NULL,
    goal TEXT NOT NULL,
    strategy TEXT NOT NULL,
    outcome TEXT NOT NULL,
    session_id TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_goal ON strategy_log(goal_hash, outcome);
CREATE INDEX IF NOT EXISTS idx_strategy_recent ON strategy_log(created_at DESC);

CREATE TABLE IF NOT EXISTS failure_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_pattern TEXT NOT NULL,
    error_type TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    last_session_id TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    UNIQUE(command_pattern, error_type)
);
"""


class LongTermMemory:
    """セッションをまたいで知識を永続化する記憶エンジン。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_LTM_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Knowledge store
    # ------------------------------------------------------------------

    def learn(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 1.0,
        session_id: Optional[str] = None,
    ) -> None:
        """知識を記憶する。既存のキーは上書き。"""
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO knowledge(key, value, confidence, session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            """,
            (key, value, confidence, session_id, now, now),
        )
        self._conn.commit()

    def recall(self, key: str) -> Optional[str]:
        """キーで記憶を取り出す。"""
        row = self._conn.execute(
            "SELECT value FROM knowledge WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def recall_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の記憶を新しい順に取り出す。"""
        rows = self._conn.execute(
            "SELECT key, value, confidence, updated_at FROM knowledge ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Strategy log
    # ------------------------------------------------------------------

    def log_strategy(
        self,
        goal: str,
        strategy: str,
        outcome: str,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """ゴールに対する戦略と結果を記録する。"""
        goal_hash = hashlib.md5(goal.encode()).hexdigest()[:8]
        self._conn.execute(
            "INSERT INTO strategy_log(goal_hash, goal, strategy, outcome, session_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (goal_hash, goal, strategy, outcome, session_id, time.time()),
        )
        self._conn.commit()

    def recall_strategies(self, goal: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        """類似ゴールでの過去の戦略を取り出す。"""
        goal_hash = hashlib.md5(goal.encode()).hexdigest()[:8]
        keywords = [w for w in goal.split() if len(w) > 2]
        keyword = keywords[0] if keywords else goal[:10]
        rows = self._conn.execute(
            """
            SELECT goal, strategy, outcome, created_at FROM strategy_log
            WHERE goal_hash = ? OR goal LIKE ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (goal_hash, f"%{keyword}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_successful_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """成功した戦略を新しい順に取り出す。"""
        rows = self._conn.execute(
            "SELECT goal, strategy, created_at FROM strategy_log WHERE outcome = 'success' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Failure log
    # ------------------------------------------------------------------

    def log_failure(
        self,
        command: str,
        error_type: str,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """失敗パターンを記録する。同じパターンはカウントアップ。"""
        now = time.time()
        pattern = command[:100]
        self._conn.execute(
            """
            INSERT INTO failure_log(command_pattern, error_type, count, last_session_id, first_seen, last_seen)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(command_pattern, error_type) DO UPDATE SET
                count = count + 1,
                last_session_id = excluded.last_session_id,
                last_seen = excluded.last_seen
            """,
            (pattern, error_type, session_id, now, now),
        )
        self._conn.commit()

    def get_known_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """既知の失敗パターンを頻度順に取り出す。"""
        rows = self._conn.execute(
            "SELECT command_pattern, error_type, count FROM failure_log ORDER BY count DESC, last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def is_known_failure(self, command: str, error_type: str) -> bool:
        """このコマンド＋エラーが2回以上失敗しているか。"""
        pattern = command[:100]
        row = self._conn.execute(
            "SELECT count FROM failure_log WHERE command_pattern = ? AND error_type = ?",
            (pattern, error_type),
        ).fetchone()
        return row is not None and row["count"] >= 2
