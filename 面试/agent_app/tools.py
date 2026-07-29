from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Type

from pydantic import BaseModel, Field, ValidationError, field_validator


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="A numeric expression using only +, -, *, /, parentheses, and decimal numbers.",
    )


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="Keywords to search in the local demonstration corpus.")


class WeatherInput(BaseModel):
    location: str = Field(min_length=1, max_length=60, description="City name, for example Beijing or Shanghai.")
    date: str | None = Field(default=None, description="Optional ISO 8601 date (YYYY-MM-DD).")

    @field_validator("date")
    @classmethod
    def require_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date must use ISO 8601 YYYY-MM-DD format.") from exc
        return value


class LoadMemoryInput(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="What kind of preference or fact is needed.")
    limit: int = Field(default=5, ge=1, le=5, description="Maximum number of memory records to return.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[..., Any]
    visible_to_model: bool = True

    def prompt_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }


class Calculator:
    _operators: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
            raise ValueError("Expression may only contain numbers, decimal points, parentheses, and + - * /.")
        try:
            root = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid arithmetic expression.") from exc

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in cls._operators:
                return cls._operators[type(node.op)](visit(node.left), visit(node.right))
            raise ValueError("Unsupported arithmetic operation.")

        result = visit(root)
        if result == float("inf") or result == float("-inf"):
            raise ValueError("Result is outside the supported range.")
        return result


SEARCH_CORPUS = [
    {
        "title": "ReAct pattern",
        "snippet": "ReAct alternates a brief decision, an action, and an observation before producing a final answer.",
        "tags": ["react", "agent", "tool", "reasoning"],
    },
    {
        "title": "SQLite sessions",
        "snippet": "A session ID isolates conversation history while SQLite provides durable local persistence.",
        "tags": ["sqlite", "session", "memory", "database"],
    },
    {
        "title": "Structured tool outputs",
        "snippet": "Stable JSON tool outputs let an agent inspect success and error cases consistently.",
        "tags": ["json", "tool", "schema", "error"],
    },
]

WEATHER_DATA = {
    "beijing": {"location": "Beijing", "condition": "Sunny", "temperature_c": 28, "humidity_percent": 35},
    "shanghai": {"location": "Shanghai", "condition": "Cloudy", "temperature_c": 27, "humidity_percent": 72},
    "shenzhen": {"location": "Shenzhen", "condition": "Light rain", "temperature_c": 30, "humidity_percent": 82},
    "hangzhou": {"location": "Hangzhou", "condition": "Partly cloudy", "temperature_c": 26, "humidity_percent": 65},
}


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def visible_definitions(self) -> list[dict[str, Any]]:
        return [spec.prompt_definition() for spec in self._specs.values() if spec.visible_to_model]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._specs.get(name)
        if not spec:
            return {"error": f"Unknown tool: {name}"}
        try:
            parsed = spec.input_model.model_validate(arguments)
            result = spec.handler(**parsed.model_dump())
            return result if isinstance(result, dict) else {"result": result}
        except ValidationError as exc:
            return {"error": "Invalid tool arguments", "details": exc.errors(include_url=False)}
        except Exception as exc:  # Tool output must always be machine-readable.
            return {"error": str(exc) or exc.__class__.__name__}


def run_calculator(expression: str) -> dict[str, Any]:
    return {"expression": expression, "result": Calculator.evaluate(expression)}


def run_search(query: str) -> dict[str, Any]:
    terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query)}
    matches = []
    for item in SEARCH_CORPUS:
        haystack = " ".join([item["title"], item["snippet"], *item["tags"]]).lower()
        if not terms or any(term in haystack for term in terms):
            matches.append({"title": item["title"], "snippet": item["snippet"]})
    return {"query": query, "source": "local_mock", "results": matches[:3]}


def run_weather(location: str, date: str | None = None) -> dict[str, Any]:
    result = WEATHER_DATA.get(location.strip().lower())
    if not result:
        return {"error": f"No mock weather is available for {location}. Try Beijing, Shanghai, Shenzhen, or Hangzhou."}
    payload = dict(result)
    payload["date"] = date or "today"
    payload["source"] = "local_mock"
    return payload


def create_default_registry(load_memory: Callable[[str, int], dict[str, Any]]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Calculate a numeric expression. Use for exact arithmetic; do not use for general reasoning or dates.",
            input_model=CalculatorInput,
            handler=run_calculator,
        )
    )
    registry.register(
        ToolSpec(
            name="search",
            description="Search the built-in demonstration knowledge corpus. Use for local project facts; do not use for live web information.",
            input_model=SearchInput,
            handler=run_search,
        )
    )
    registry.register(
        ToolSpec(
            name="weather",
            description="Get mock weather for Beijing, Shanghai, Shenzhen, or Hangzhou. Use only when a weather answer is needed.",
            input_model=WeatherInput,
            handler=run_weather,
        )
    )
    registry.register(
        ToolSpec(
            name="load_memory",
            description="Load up to five relevant local preferences or facts. Use when the automatically injected memory is insufficient.",
            input_model=LoadMemoryInput,
            handler=load_memory,
        )
    )
    return registry


def pretty_tool_definitions(registry: ToolRegistry) -> str:
    return json.dumps(registry.visible_definitions(), ensure_ascii=False, indent=2)
