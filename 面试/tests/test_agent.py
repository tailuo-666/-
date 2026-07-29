from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_app.agent import ReActAgent, parse_react_output
from agent_app.config import Settings
from agent_app.llm import ContextOverflowError
from agent_app.memory import MemoryCandidate, MemoryManager
from agent_app.store import Store
from agent_app.tools import create_default_registry


class FakeLLM:
    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_output_tokens: int = 1200) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return "[]" if "Extract only durable" in prompt else "Thought: complete\nFinal Answer: done"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_agent(root: Path, responses: list[str | Exception], *, max_steps: int = 6) -> tuple[ReActAgent, Store, FakeLLM]:
    settings = Settings(
        data_dir=root,
        database_path=root / "agent.sqlite3",
        memory_dir=root / "memories",
        openai_api_key=None,
        openai_base_url="http://example.invalid",
        openai_model="test-model",
        max_steps=max_steps,
    )
    store = Store(settings.database_path)
    memory = MemoryManager(settings.memory_dir)
    llm = FakeLLM(responses)
    return ReActAgent(settings, store, memory, llm, create_default_registry(memory.load_tool)), store, llm


class ReActTests(unittest.TestCase):
    def test_parser_accepts_action_and_final(self) -> None:
        action = parse_react_output('Thought: Need exact arithmetic.\nAction: calculator\nAction Input: {"expression": "2+2"}')
        self.assertEqual(action.action, "calculator")
        self.assertEqual(action.arguments, {"expression": "2+2"})
        final = parse_react_output("Thought: I have the result.\nFinal Answer: 4")
        self.assertEqual(final.answer, "4")

    def test_multistep_tool_call_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent, store, _ = make_agent(
                root,
                [
                    'Thought: Calculate exactly.\nAction: calculator\nAction Input: {"expression": "2 + 3"}',
                    "Thought: The calculation is complete.\nFinal Answer: The result is 5.",
                    "[]",
                ],
            )
            session = store.create_session()
            assistant, trace = agent.create_turn(session["id"], "What is 2 + 3?")
            completed = agent.complete_turn(assistant["id"])
            self.assertEqual(completed["content"], "The result is 5.")
            spans = store.list_spans(trace["id"], kind="tool")
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["name"], "calculator")
            self.assertIsNotNone(spans[0]["parent_span_id"])
            self.assertEqual(json.loads(spans[0]["output_json"])["result"], 5.0)

    def test_bad_protocol_becomes_an_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent, store, _ = make_agent(
                Path(temporary),
                ["This is not ReAct output.", "Thought: Recover.\nFinal Answer: Recovered.", "[]"],
            )
            session = store.create_session()
            assistant, trace = agent.create_turn(session["id"], "hello")
            completed = agent.complete_turn(assistant["id"])
            self.assertEqual(completed["content"], "Recovered.")
            trace_data = store.get_trace(trace["id"])
            trajectory = json.loads(trace_data["trajectory_json"])
            self.assertEqual(trajectory[0]["action"], "parser_error")

    def test_max_steps_forces_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent, store, _ = make_agent(
                Path(temporary),
                [
                    'Thought: Need a tool.\nAction: calculator\nAction Input: {"expression": "1+1"}',
                    "The best answer is 2.",
                    "[]",
                ],
                max_steps=1,
            )
            session = store.create_session()
            assistant, _ = agent.create_turn(session["id"], "one plus one")
            completed = agent.complete_turn(assistant["id"])
            self.assertEqual(completed["content"], "The best answer is 2.")

    def test_resume_reuses_completed_tool_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent, store, _ = make_agent(
                Path(temporary),
                ["Thought: The saved observation is enough.\nFinal Answer: It was 4.", "[]"],
            )
            session = store.create_session()
            assistant, trace = agent.create_turn(session["id"], "What was 2+2?")
            span = store.start_span(trace["id"], kind="tool", name="calculator", input_data={"expression": "2+2"})
            store.finish_span(span, output={"expression": "2+2", "result": 4})
            completed = agent.complete_turn(assistant["id"])
            self.assertEqual(completed["content"], "It was 4.")
            self.assertEqual(len(store.list_spans(trace["id"], kind="tool")), 1)

    def test_context_overflow_retries_after_compression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent, store, _ = make_agent(
                Path(temporary),
                [
                    ContextOverflowError("too large"),
                    "facts: user asked a question",
                    "Thought: Compact context worked.\nFinal Answer: Done after compression.",
                    "[]",
                ],
            )
            session = store.create_session()
            assistant, trace = agent.create_turn(session["id"], "hello")
            completed = agent.complete_turn(assistant["id"])
            self.assertEqual(completed["content"], "Done after compression.")
            span_names = [span["name"] for span in store.list_spans(trace["id"])]
            self.assertIn("emergency_compression", span_names)


class MemoryTests(unittest.TestCase):
    def test_memory_index_fallback_and_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = MemoryManager(Path(temporary) / "memories", consolidation_threshold=10)
            self.assertTrue(
                memory.write_candidate(
                    MemoryCandidate(
                        name="Tabs for indentation",
                        type="preference",
                        description="The user prefers tabs for indentation.",
                        body="Use tabs when editing project files.",
                    )
                )
            )
            self.assertFalse(
                memory.write_candidate(
                    MemoryCandidate(
                        name="Tabs for indentation",
                        type="preference",
                        description="The user prefers tabs for indentation.",
                        body="Use tabs when editing project files.",
                    )
                )
            )
            self.assertIn("Tabs for indentation", (memory.memory_dir / "MEMORY.md").read_text(encoding="utf-8"))
            loaded = memory.load_tool("indentation tabs")
            self.assertEqual(loaded["memories"][0]["type"], "preference")
            llm = FakeLLM(
                [
                    '[{"name":"Local mode","type":"fact","description":"The app is local only.","body":"Run it on the local machine."}]'
                ]
            )
            self.assertEqual(memory.extract_from_messages([{"role": "user", "content": "Keep this app local."}], llm), 1)
            self.assertEqual(len(memory.list_records()), 2)

    def test_memory_consolidation_replaces_records_after_valid_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = MemoryManager(Path(temporary) / "memories", consolidation_threshold=2)
            memory.write_candidate(MemoryCandidate(name="A", type="fact", description="First fact", body="First body"))
            memory.write_candidate(MemoryCandidate(name="B", type="fact", description="Second fact", body="Second body"))
            llm = FakeLLM(
                ['[{"name":"Merged","type":"fact","description":"Merged fact","body":"Combined body"}]']
            )
            self.assertTrue(memory.consolidate(llm))
            records = memory.list_records()
            self.assertEqual([record.name for record in records], ["Merged"])
