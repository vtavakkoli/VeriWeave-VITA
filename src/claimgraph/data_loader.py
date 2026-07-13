from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import file_sha256


@dataclass(frozen=True)
class PolicyDocument:
    source: str
    title: str
    text: str
    version: str
    domain: str
    sha256: str
    path: str


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    task_type: str
    question: str
    expected_answer: str
    expected_policy_refs: list[str]
    expected_decision: str
    expected_human_review: bool | None = None
    policy_version: str = ""
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


def _header_value(text: str, name: str, default: str = "") -> str:
    match = re.search(rf"(?im)^{re.escape(name)}\s*:\s*([^\n]+)", text)
    return match.group(1).strip() if match else default


def load_policies(data_dir: Path) -> list[PolicyDocument]:
    docs: list[PolicyDocument] = []
    for path in sorted((data_dir / "policies").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"(?m)^#\s+(.+)$", text)
        version = _header_value(text, "version", "v1")
        domain = _header_value(text, "domain", path.stem.replace("_", "-"))
        docs.append(
            PolicyDocument(
                source=path.name,
                title=title_match.group(1).strip() if title_match else path.stem,
                text=text,
                version=version,
                domain=domain,
                sha256=file_sha256(path),
                path=str(path),
            )
        )
    return docs


def load_tasks(path: Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(BenchmarkTask(**raw))
    return tasks
