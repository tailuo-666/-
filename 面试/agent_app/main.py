from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import ReActAgent
from .config import PROJECT_ROOT, Settings
from .llm import LLMClient, OpenAITextClient
from .memory import MemoryManager
from .schemas import (
    AgentReply,
    CreateSessionRequest,
    MessageView,
    SendMessageRequest,
    SessionDetail,
    SessionView,
    ToolSpanView,
    TraceView,
)
from .store import Store, parse_time
from .tools import create_default_registry


STATIC_DIR = PROJECT_ROOT / "static"


def _message_view(message: dict[str, Any]) -> MessageView:
    return MessageView(
        id=message["id"],
        role=message["role"],
        content=message["content"],
        quoted_text=message["quoted_text"],
        status=message["status"],
        trace_id=message["trace_id"],
        created_at=parse_time(message["created_at"]),
    )


def _session_view(session: dict[str, Any]) -> SessionView:
    return SessionView(
        id=session["id"],
        title=session["title"],
        status=session["status"],
        created_at=parse_time(session["created_at"]),
        updated_at=parse_time(session["updated_at"]),
    )


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()
    store = Store(resolved_settings.database_path)
    memory = MemoryManager(resolved_settings.memory_dir)
    resolved_llm = llm or OpenAITextClient(resolved_settings)
    registry = create_default_registry(memory.load_tool)
    agent = ReActAgent(resolved_settings, store, memory, resolved_llm, registry)

    app = FastAPI(title="ReAct Workbench", version="0.1.0")
    app.state.settings = resolved_settings
    app.state.store = store
    app.state.memory = memory
    app.state.agent = agent
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_configured": bool(resolved_settings.openai_api_key)}

    @app.post("/api/sessions", response_model=SessionView)
    def create_session(request: CreateSessionRequest) -> SessionView:
        return _session_view(store.create_session(request.title))

    @app.get("/api/sessions", response_model=list[SessionView])
    def list_sessions() -> list[SessionView]:
        return [_session_view(session) for session in store.list_sessions()]

    @app.get("/api/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str) -> SessionDetail:
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        pending = store.find_pending_assistant_message(session_id)
        return SessionDetail(
            **_session_view(session).model_dump(),
            messages=[_message_view(message) for message in store.list_messages(session_id)],
            pending_message_id=pending["id"] if pending else None,
        )

    @app.post("/api/sessions/{session_id}/messages", response_model=AgentReply)
    def send_message(session_id: str, request: SendMessageRequest) -> AgentReply:
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        try:
            assistant_message, trace = agent.create_turn(session_id, request.content, request.quoted_text)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        completed = agent.complete_turn(assistant_message["id"])
        return AgentReply(message=_message_view(completed), trace_id=trace["id"])

    @app.post("/api/sessions/{session_id}/resume", response_model=AgentReply)
    def resume_session(session_id: str) -> AgentReply:
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        pending = store.find_pending_assistant_message(session_id)
        if not pending:
            raise HTTPException(status_code=409, detail="There is no running request to resume.")
        completed = agent.complete_turn(pending["id"])
        return AgentReply(message=_message_view(completed), trace_id=completed["trace_id"])

    @app.get("/api/traces/{trace_id}", response_model=TraceView)
    def get_trace(trace_id: str) -> TraceView:
        trace = store.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found.")
        tool_spans = []
        for span in store.list_spans(trace_id, kind="tool"):
            tool_spans.append(
                ToolSpanView(
                    id=span["id"],
                    name=span["name"],
                    status=span["status"],
                    input=json.loads(span["input_json"]) if span["input_json"] else None,
                    output=json.loads(span["output_json"]) if span["output_json"] else None,
                    error=span["error"],
                    started_at=parse_time(span["started_at"]),
                    finished_at=parse_time(span["finished_at"]) if span["finished_at"] else None,
                )
            )
        return TraceView(
            id=trace["id"],
            session_id=trace["session_id"],
            message_id=trace["message_id"],
            status=trace["status"],
            started_at=parse_time(trace["started_at"]),
            finished_at=parse_time(trace["finished_at"]) if trace["finished_at"] else None,
            tool_spans=tool_spans,
        )

    return app


app = create_app()
