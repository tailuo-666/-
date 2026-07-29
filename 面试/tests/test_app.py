from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from agent_app.config import Settings
from agent_app.main import create_app
from tests.test_agent import FakeLLM


class ApiTests(unittest.TestCase):
    def test_session_message_and_trace_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                data_dir=root,
                database_path=root / "agent.sqlite3",
                memory_dir=root / "memories",
                openai_api_key=None,
                openai_base_url="http://example.invalid",
                openai_model="test-model",
            )
            llm = FakeLLM(
                [
                    'Thought: Calculate it.\nAction: calculator\nAction Input: {"expression": "7*6"}',
                    "Thought: I have the result.\nFinal Answer: 42",
                    "[]",
                ]
            )
            client = TestClient(create_app(settings, llm))
            created = client.post("/api/sessions", json={"title": "Arithmetic"})
            self.assertEqual(created.status_code, 200)
            session_id = created.json()["id"]
            response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "What is 7*6?"})
            self.assertEqual(response.status_code, 200)
            reply = response.json()
            self.assertEqual(reply["message"]["content"], "42")
            trace = client.get(f"/api/traces/{reply['trace_id']}")
            self.assertEqual(trace.status_code, 200)
            self.assertEqual(trace.json()["tool_spans"][0]["name"], "calculator")
            self.assertNotIn("thought", trace.json())
