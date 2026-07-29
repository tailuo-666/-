from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    memory_dir: Path
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    max_steps: int = 6
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls, data_dir: Path | None = None) -> "Settings":
        resolved_data_dir = data_dir or Path(os.getenv("AGENT_DATA_DIR", PROJECT_ROOT / "data"))
        return cls(
            data_dir=resolved_data_dir,
            database_path=resolved_data_dir / "agent.sqlite3",
            memory_dir=resolved_data_dir / "memories",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.longxiadev.store/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        )
