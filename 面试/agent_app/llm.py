from __future__ import annotations

import random
import threading
import time
from collections import deque
from typing import Any, Protocol

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


class ContextOverflowError(LLMError):
    pass


class LLMClient(Protocol):
    def generate(self, prompt: str, *, max_output_tokens: int = 1200) -> str: ...


class OpenAITextClient:
    """OpenAI-compatible Responses client with the wire format accepted by the configured gateway."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = (
            httpx.Client(
                base_url=settings.openai_base_url.rstrip("/"),
                timeout=settings.request_timeout_seconds,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            if settings.openai_api_key
            else None
        )
        self._failures: deque[float] = deque()
        self._circuit_open_until = 0.0
        self._lock = threading.Lock()

    def _record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._failures.append(now)
            while self._failures and now - self._failures[0] > 60:
                self._failures.popleft()
            if len(self._failures) >= 3:
                self._circuit_open_until = now + 60

    def generate(self, prompt: str, *, max_output_tokens: int = 1200) -> str:
        if not self._client:
            raise LLMError("OPENAI_API_KEY is not configured. Set it in the process environment before sending a message.")
        with self._lock:
            if time.monotonic() < self._circuit_open_until:
                raise LLMError("The model circuit breaker is open after repeated failures. Try again in one minute.")

        for attempt in range(3):
            try:
                response = self._client.post(
                    "/responses",
                    json={
                        "model": self.settings.openai_model,
                        "input": prompt,
                        "max_output_tokens": max_output_tokens,
                        "store": False,
                    },
                )
                if response.status_code == 413:
                    raise ContextOverflowError("The model rejected the prompt as too large.")
                response.raise_for_status()
                return self._extract_output_text(response.json())
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code == 429 or 500 <= status_code < 600
                if not retryable or attempt == 2:
                    self._record_failure()
                    raise LLMError(f"Model request failed with HTTP {status_code}.") from exc
            except httpx.HTTPError as exc:
                if attempt == 2:
                    self._record_failure()
                    raise LLMError("Could not connect to the model service.") from exc
            time.sleep((0.5 * (2**attempt)) + random.random() * 0.15)
        raise LLMError("Model request failed.")

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        direct_output = payload.get("output_text")
        if isinstance(direct_output, str) and direct_output.strip():
            return direct_output.strip()
        text_parts: list[str] = []
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
        result = "".join(text_parts).strip()
        if not result:
            raise LLMError("Model response did not contain output text.")
        return result
