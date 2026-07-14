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
    authority_rank: int = 50


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
    documents: list[PolicyDocument] = []
    for path in sorted((data_dir / "policies").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"(?m)^#\s+(.+)$", text)
        version = _header_value(text, "version", "v1")
        domain = _header_value(text, "domain", path.stem.replace("_", "-"))
        raw_rank = _header_value(text, "authority_rank", "50")
        try:
            authority_rank = int(raw_rank)
        except ValueError:
            authority_rank = 50
        documents.append(
            PolicyDocument(
                source=path.name,
                title=title_match.group(1).strip() if title_match else path.stem,
                text=text,
                version=version,
                domain=domain,
                sha256=file_sha256(path),
                path=str(path),
                authority_rank=authority_rank,
            )
        )
    return documents


def load_tasks(path: Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(BenchmarkTask(**json.loads(line)))
    return tasks
