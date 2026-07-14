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
    benchmark_file: str = os.getenv("BENCHMARK_FILE", "data/tasks/veriweave_benchmark.jsonl")
    graph_export_file: str = os.getenv("GRAPH_EXPORT_FILE", "data/graph/evidence_graph.json")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    healthcheck_seconds: float = float(os.getenv("OLLAMA_HEALTHCHECK_SECONDS", "2"))
    ollama_enabled: bool = _bool_env("OLLAMA_ENABLED", True)
    seed: int = int(os.getenv("SEED", "7"))
    max_tasks: int = int(os.getenv("MAX_TASKS", "0"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    vita_max_candidates: int = int(os.getenv("VITA_MAX_CANDIDATES", "8"))
    vita_max_coalition_size: int = int(os.getenv("VITA_MAX_COALITION_SIZE", "2"))
    vita_max_rounds: int = int(os.getenv("VITA_MAX_ROUNDS", "3"))
    vita_max_expand_per_round: int = int(os.getenv("VITA_MAX_EXPAND_PER_ROUND", "4"))
    boltzmann_max_candidates: int = int(os.getenv("BOLTZMANN_MAX_CANDIDATES", "12"))
    boltzmann_temperature: float = float(os.getenv("BOLTZMANN_TEMPERATURE", "0.75"))
    boltzmann_min_temperature: float = float(os.getenv("BOLTZMANN_MIN_TEMPERATURE", "0.20"))
    boltzmann_max_temperature: float = float(os.getenv("BOLTZMANN_MAX_TEMPERATURE", "1.50"))
    boltzmann_attention_mass: float = float(os.getenv("BOLTZMANN_ATTENTION_MASS", "0.90"))
    boltzmann_anneal_rate: float = float(os.getenv("BOLTZMANN_ANNEAL_RATE", "0.82"))
    boltzmann_size_penalty: float = float(os.getenv("BOLTZMANN_SIZE_PENALTY", "0.08"))
    methods: str = os.getenv(
        "METHODS",
        "Direct LLM,Text RAG,Community GraphRAG,PPR GraphRAG,Steiner GraphRAG,VeriWeave-Core,VeriWeave-Horizon,VeriWeave-VITA,VeriWeave-VITA-BPA",
    )

    def method_list(self) -> list[str]:
        allowed = {
            "Direct LLM",
            "Text RAG",
            "Community GraphRAG",
            "PPR GraphRAG",
            "Steiner GraphRAG",
            "VeriWeave-Core",
            "VeriWeave-Horizon",
            "VeriWeave-VITA",
            "VeriWeave-VITA-BPA",
        }
        result: list[str] = []
        for item in self.methods.split(","):
            name = item.strip()
            if name in allowed and name not in result:
                result.append(name)
        return result or ["Direct LLM", "Text RAG", "Community GraphRAG", "PPR GraphRAG", "Steiner GraphRAG", "VeriWeave-Core", "VeriWeave-Horizon", "VeriWeave-VITA", "VeriWeave-VITA-BPA"]
