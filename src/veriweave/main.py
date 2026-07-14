from __future__ import annotations

import csv
import json
import logging
import platform
import random
from collections import Counter
from pathlib import Path

from .agents import run_method
from .audit import attach_independent_audit
from .config import Config
from .data_loader import load_policies, load_tasks
from .evaluators import evaluate
from .graph import build_property_graph
from .ollama_client import OllamaClient
from .report_generator import generate_report
from .retrieval import (
    AuditRetriever,
    CommunityGraphRetriever,
    PPRGraphRetriever,
    SteinerGraphRetriever,
    TextRetriever,
    VeriWeaveRetriever,
)
from .stats import write_statistics
from .utils import append_jsonl, now, stable_hash


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("veriweave")

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
    community_retriever = CommunityGraphRetriever(graph)
    ppr_retriever = PPRGraphRetriever(graph)
    steiner_retriever = SteinerGraphRetriever(graph)
    veriweave_retriever = VeriWeaveRetriever(graph)
    audit_retriever = AuditRetriever(graph)
    client = OllamaClient(
        cfg.ollama_base_url,
        cfg.ollama_model,
        cfg.timeout_seconds,
        trace_path,
        cfg.ollama_enabled,
        cfg.healthcheck_seconds,
    )

    manifest = {
        "timestamp": now(),
        "run_id": stable_hash(now() + cfg.ollama_model + str(cfg.seed)),
        "system": "VeriWeave-VITA-BPA",
        "version": "4.0.0",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model": cfg.ollama_model,
        "seed": cfg.seed,
        "ollama_enabled": cfg.ollama_enabled,
        "ollama_available": client.is_available(),
        "benchmark": str(benchmark_path),
        "tasks_evaluated": len(tasks),
        "task_type_distribution": dict(Counter(task.task_type for task in tasks)),
        "methods": methods,
        "method_notes": {
            "Community GraphRAG": "concept-community baseline inspired by GraphRAG; not the official implementation",
            "PPR GraphRAG": "Personalized-PageRank baseline inspired by HippoRAG; not the official implementation",
            "Steiner GraphRAG": "connected-subgraph heuristic inspired by G-Retriever; not a faithful PCST/GNN reproduction",
            "VeriWeave-Core": "support/counterevidence retrieval, precedence resolution, independent claim validation, and claim-selective decision gating",
            "VeriWeave-Horizon": "VeriWeave-Core plus singleton counterfactual evidence-horizon search, decision-impact testing, and minimal-evidence-cut certification",
            "VeriWeave-VITA": "VeriWeave-Horizon plus coalitional omitted-evidence closure, a two-sided decision space, grounded argumentation, and temporal decision-drift certification",
            "VeriWeave-VITA-BPA": "VeriWeave-VITA plus inference-time Boltzmann Policy Attention over policy-clause coalitions, adaptive temperature, pairwise interaction couplings, and free-energy residual certification",
        },
        "vita_budget": {
            "max_candidates": cfg.vita_max_candidates,
            "max_coalition_size": cfg.vita_max_coalition_size,
            "max_rounds": cfg.vita_max_rounds,
            "max_expand_per_round": cfg.vita_max_expand_per_round,
        },
        "boltzmann_policy_attention": {
            "max_candidates": cfg.boltzmann_max_candidates,
            "base_temperature": cfg.boltzmann_temperature,
            "minimum_temperature": cfg.boltzmann_min_temperature,
            "maximum_temperature": cfg.boltzmann_max_temperature,
            "attention_mass": cfg.boltzmann_attention_mass,
            "anneal_rate": cfg.boltzmann_anneal_rate,
            "coalition_size_penalty": cfg.boltzmann_size_penalty,
            "exact_enumeration": True,
            "token_attention": False,
        },
        "evaluation": {
            "common_audit_for_all_methods": True,
            "auto_citations": False,
            "model_decision_preserved": True,
            "compatibility_score_reported_separately": True,
            "common_counterfactual_horizon_audit": True,
            "robustness_in_overall_score": True,
            "coalitional_vita_audit": True,
            "argumentation_certificate": True,
            "temporal_drift_certificate": True,
            "common_boltzmann_policy_attention_audit": True,
        },
        "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges), "export": str(graph_path)},
        "note": "Offline fallback outputs verify software only and must not be reported as LLM experimental findings.",
    }
    (cfg.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict] = []
    traces: list[dict] = []
    failures: list[dict] = []
    for index, task in enumerate(tasks, start=1):
        logger.info("Task %s/%s: %s", index, len(tasks), task.id)
        for method in methods:
            try:
                trace = run_method(
                    method,
                    task,
                    graph,
                    text_retriever,
                    community_retriever,
                    ppr_retriever,
                    steiner_retriever,
                    veriweave_retriever,
                    client,
                    cfg.top_k,
                    {
                        "max_candidates": cfg.vita_max_candidates,
                        "max_coalition_size": cfg.vita_max_coalition_size,
                        "max_rounds": cfg.vita_max_rounds,
                        "max_expand_per_round": cfg.vita_max_expand_per_round,
                        "boltzmann_max_candidates": cfg.boltzmann_max_candidates,
                        "boltzmann_temperature": cfg.boltzmann_temperature,
                        "boltzmann_min_temperature": cfg.boltzmann_min_temperature,
                        "boltzmann_max_temperature": cfg.boltzmann_max_temperature,
                        "boltzmann_attention_mass": cfg.boltzmann_attention_mass,
                        "boltzmann_anneal_rate": cfg.boltzmann_anneal_rate,
                        "boltzmann_size_penalty": cfg.boltzmann_size_penalty,
                    },
                )
                attach_independent_audit(trace, graph, audit_retriever, max(10, cfg.top_k + 4))
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
    statistics = write_statistics(cfg.result_dir, rows, target="VeriWeave-VITA-BPA", seed=cfg.seed)
    generate_report(cfg.result_dir, rows, traces, manifest, statistics)
    logger.info("Report written to %s", cfg.result_dir / "report.html")


if __name__ == "__main__":
    main()
