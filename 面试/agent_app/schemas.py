from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12_000)
    quoted_text: str | None = Field(default=None, max_length=8_000)


class MessageView(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    quoted_text: str | None = None
    status: str
    trace_id: str | None = None
    created_at: datetime


class SessionView(BaseModel):
    id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionView):
    messages: list[MessageView]
    pending_message_id: str | None = None


class ToolSpanView(BaseModel):
    id: str
    name: str
    status: str
    input: dict[str, Any] | None = None
    output: Any | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class TraceView(BaseModel):
    id: str
    session_id: str
    message_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    tool_spans: list[ToolSpanView]


class AgentReply(BaseModel):
    message: MessageView
    trace_id: str
