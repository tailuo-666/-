from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Store:
    """SQLite persistence with short-lived connections for FastAPI worker threads."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    quoted_text TEXT,
                    status TEXT NOT NULL,
                    trace_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    trajectory_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS spans (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                    parent_span_id TEXT,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_spans_trace_started
                ON spans(trace_id, started_at);
                """
            )

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = utc_now()
        session = {
            "id": session_id,
            "title": title.strip() if title and title.strip() else "新对话",
            "status": "idle",
            "summary": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._write_lock, self.connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, status, summary, created_at, updated_at) "
                "VALUES (:id, :title, :status, :summary, :created_at, :updated_at)",
                session,
            )
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        summary: str | None = None,
        title: str | None = None,
    ) -> None:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        values.append(session_id)
        with self._write_lock, self.connection() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", values)

    def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        status: str,
        quoted_text: str | None = None,
        trace_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        message = {
            "id": message_id or str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "quoted_text": quoted_text,
            "status": status,
            "trace_id": trace_id,
            "created_at": utc_now(),
        }
        with self._write_lock, self.connection() as conn:
            conn.execute(
                """INSERT INTO messages
                (id, session_id, role, content, quoted_text, status, trace_id, created_at)
                VALUES (:id, :session_id, :role, :content, :quoted_text, :status, :trace_id, :created_at)""",
                message,
            )
        return message

    def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        status: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if content is not None:
            fields.append("content = ?")
            values.append(content)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if trace_id is not None:
            fields.append("trace_id = ?")
            values.append(trace_id)
        if not fields:
            return
        values.append(message_id)
        with self._write_lock, self.connection() as conn:
            conn.execute(f"UPDATE messages SET {', '.join(fields)} WHERE id = ?", values)

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return dict(row) if row else None

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def find_pending_assistant_message(self, session_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT * FROM messages WHERE session_id = ? AND role = 'assistant'
                AND status = 'running' ORDER BY created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_trace(self, session_id: str, message_id: str) -> dict[str, Any]:
        trace = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "message_id": message_id,
            "status": "running",
            "trajectory_json": "[]",
            "started_at": utc_now(),
            "finished_at": None,
        }
        with self._write_lock, self.connection() as conn:
            conn.execute(
                """INSERT INTO traces
                (id, session_id, message_id, status, trajectory_json, started_at, finished_at)
                VALUES (:id, :session_id, :message_id, :status, :trajectory_json, :started_at, :finished_at)""",
                trace,
            )
        return trace

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        return dict(row) if row else None

    def update_trace(self, trace_id: str, *, status: str | None = None, trajectory: list[dict[str, Any]] | None = None) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
            if status in {"completed", "failed"}:
                fields.append("finished_at = ?")
                values.append(utc_now())
        if trajectory is not None:
            fields.append("trajectory_json = ?")
            values.append(json.dumps(trajectory, ensure_ascii=False))
        if not fields:
            return
        values.append(trace_id)
        with self._write_lock, self.connection() as conn:
            conn.execute(f"UPDATE traces SET {', '.join(fields)} WHERE id = ?", values)

    def start_span(
        self,
        trace_id: str,
        *,
        kind: str,
        name: str,
        input_data: Any = None,
        parent_span_id: str | None = None,
    ) -> str:
        span_id = str(uuid.uuid4())
        with self._write_lock, self.connection() as conn:
            conn.execute(
                """INSERT INTO spans
                (id, trace_id, parent_span_id, kind, name, status, input_json, started_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (
                    span_id,
                    trace_id,
                    parent_span_id,
                    kind,
                    name,
                    json.dumps(input_data, ensure_ascii=False) if input_data is not None else None,
                    utc_now(),
                ),
            )
        return span_id

    def finish_span(self, span_id: str, *, output: Any = None, error: str | None = None) -> None:
        status = "failed" if error else "completed"
        with self._write_lock, self.connection() as conn:
            conn.execute(
                "UPDATE spans SET status = ?, output_json = ?, error = ?, finished_at = ? WHERE id = ?",
                (
                    status,
                    json.dumps(output, ensure_ascii=False) if output is not None else None,
                    error,
                    utc_now(),
                    span_id,
                ),
            )

    def list_spans(self, trace_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        clause = "WHERE trace_id = ?"
        values: list[Any] = [trace_id]
        if kind:
            clause += " AND kind = ?"
            values.append(kind)
        with self.connection() as conn:
            rows = conn.execute(f"SELECT * FROM spans {clause} ORDER BY started_at ASC", values).fetchall()
        return [dict(row) for row in rows]

    def trace_tool_observations(self, trace_id: str) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for span in self.list_spans(trace_id, kind="tool"):
            if span["status"] != "completed":
                continue
            input_data = json.loads(span["input_json"]) if span["input_json"] else {}
            output = json.loads(span["output_json"]) if span["output_json"] else None
            observations.append({"action": span["name"], "input": input_data, "observation": output})
        return observations
