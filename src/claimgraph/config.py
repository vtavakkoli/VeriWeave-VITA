from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    result_dir: Path = root / "result"
    benchmark_file: str = os.getenv("BENCHMARK_FILE", "data/tasks/claimgraph_benchmark.jsonl")
    graph_export_file: str = os.getenv("GRAPH_EXPORT_FILE", "data/graph/property_graph.json")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    healthcheck_seconds: float = float(os.getenv("OLLAMA_HEALTHCHECK_SECONDS", "2"))
    ollama_enabled: bool = _bool_env("OLLAMA_ENABLED", True)
    seed: int = int(os.getenv("SEED", "7"))
    max_tasks: int = int(os.getenv("MAX_TASKS", "0"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    methods: str = os.getenv("METHODS", "Direct LLM,Text RAG,Graph RAG,ClaimGraph")

    def method_list(self) -> list[str]:
        allowed = {"Direct LLM", "Text RAG", "Graph RAG", "ClaimGraph"}
        methods: list[str] = []
        for item in self.methods.split(","):
            item = item.strip()
            if item in allowed and item not in methods:
                methods.append(item)
        return methods or ["Direct LLM", "Text RAG", "Graph RAG", "ClaimGraph"]
