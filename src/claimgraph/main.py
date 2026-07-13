from __future__ import annotations

import csv
import json
import logging
import platform
import random
from collections import Counter
from pathlib import Path

from .agents import run_method
from .config import Config
from .data_loader import load_policies, load_tasks
from .evaluators import evaluate
from .graph import build_property_graph
from .ollama_client import OllamaClient
from .report_generator import generate_report
from .retrieval import GraphRetriever, TextRetriever
from .stats import write_statistics
from .utils import append_jsonl, now, stable_hash


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("claimgraph")

    policies = load_policies(cfg.data_dir)
    graph = build_property_graph(policies)
    graph_path = Path(cfg.graph_export_file)
    if not graph_path.is_absolute():
        graph_path = cfg.root / graph_path
    graph.export_json(graph_path)

    benchmark_path = Path(cfg.benchmark_file)
    if not benchmark_path.is_absolute():
        benchmark_path = cfg.root / benchmark_path
    tasks = load_tasks(benchmark_path)
    if cfg.max_tasks > 0:
        tasks = tasks[:cfg.max_tasks]

    methods = cfg.method_list()
    trace_path = cfg.result_dir / "traces.jsonl"
    trace_path.write_text("", encoding="utf-8")
    text_retriever = TextRetriever(graph)
    graph_retriever = GraphRetriever(graph)
    client = OllamaClient(cfg.ollama_base_url, cfg.ollama_model, cfg.timeout_seconds, trace_path, cfg.ollama_enabled, cfg.healthcheck_seconds)

    manifest = {
        "timestamp": now(),
        "run_id": stable_hash(now() + cfg.ollama_model + str(cfg.seed)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model": cfg.ollama_model,
        "ollama_enabled": cfg.ollama_enabled,
        "ollama_available": client.is_available(),
        "benchmark": str(benchmark_path),
        "tasks_evaluated": len(tasks),
        "task_type_distribution": dict(Counter(task.task_type for task in tasks)),
        "methods": methods,
        "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges), "export": str(graph_path)},
        "note": "Offline fallback outputs are for software verification only and must not be reported as experimental findings.",
    }
    (cfg.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict] = []
    traces: list[dict] = []
    failures: list[dict] = []
    for index, task in enumerate(tasks, start=1):
        logger.info("Task %s/%s: %s", index, len(tasks), task.id)
        for method in methods:
            try:
                trace = run_method(method, task, graph, text_retriever, graph_retriever, client, cfg.top_k)
                traces.append(trace)
                append_jsonl(trace_path, {"kind": "decision_trace", **trace})
                rows.append({
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "difficulty": task.difficulty,
                    "method": method,
                    **evaluate(task, trace),
                })
            except Exception as exc:
                logger.exception("Failure task=%s method=%s", task.id, method)
                failures.append({"task_id": task.id, "method": method, "error": str(exc)})

    _write_csv(cfg.result_dir / "metrics.csv", rows)
    (cfg.result_dir / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.result_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    write_statistics(cfg.result_dir, rows)
    generate_report(cfg.result_dir, rows, traces, manifest)
    logger.info("Report written to %s", cfg.result_dir / "report.html")


if __name__ == "__main__":
    main()
