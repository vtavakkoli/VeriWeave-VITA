from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .data_loader import load_policies
from .envelope import build_verification_envelope
from .graph import build_property_graph
from .models import CandidateResponse
from .retrieval import evidence_from_node
from .utils import mean


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_challenge(root: Path, challenge_file: Path, output_dir: Path) -> dict:
    graph = build_property_graph(load_policies(root / "data"))
    by_citation = {
        str(node.properties.get("citation_id")): node
        for node in graph.nodes.values()
        if node.type == "Clause"
    }
    rows: list[dict] = []
    for task in _load_jsonl(challenge_file):
        visible = []
        for citation in task["visible_citations"]:
            node = by_citation.get(citation)
            if node is None:
                raise ValueError(f"Unknown visible citation in {task['id']}: {citation}")
            visible.append(
                evidence_from_node(
                    node,
                    0.9,
                    [node.id],
                    role="visible-legacy",
                    retriever="Horizon challenge",
                )
            )
        candidate = CandidateResponse(
            answer=task["candidate_answer"],
            decision=task["candidate_decision"],
            citations=list(task["visible_citations"]),
            human_review_required=False,
        )
        core = build_verification_envelope(
            candidate,
            visible,
            graph,
            question=task["question"],
            enable_horizon=False,
        )
        horizon = build_verification_envelope(
            candidate,
            visible,
            graph,
            question=task["question"],
            enable_horizon=True,
        )
        certificate = horizon.evidence_horizon or {}
        blind_spots = certificate.get("decision_changing_blind_spots", [])
        ranked_citations = [item.get("citation_id") for item in blind_spots]
        hidden = task["designated_hidden_citation"]
        hidden_rank = ranked_citations.index(hidden) + 1 if hidden in ranked_citations else 0
        rows.append({
            "task_id": task["id"],
            "family": task["family"],
            "core_decision": core.gated_decision,
            "horizon_decision": horizon.gated_decision,
            "expected_core_decision": task["expected_core_decision"],
            "expected_horizon_decision": task["expected_horizon_decision"],
            "core_correct": int(core.gated_decision == task["expected_core_decision"]),
            "horizon_correct": int(horizon.gated_decision == task["expected_horizon_decision"]),
            "designated_hidden_found": int(hidden_rank > 0),
            "designated_hidden_rank": hidden_rank,
            "decision_changing_blind_spots": len(blind_spots),
            "additional_risk_elevating_clauses": max(0, len(blind_spots) - int(hidden_rank > 0)),
            "horizon_stability": float(certificate.get("horizon_stability", 0.0)),
            "evidence_cut_robustness": float(certificate.get("evidence_cut_robustness", 0.0)),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (output_dir / "horizon_challenge.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    found = [row["designated_hidden_found"] for row in rows]
    ranks = [row["designated_hidden_rank"] for row in rows if row["designated_hidden_rank"] > 0]
    summary = {
        "tasks": len(rows),
        "families": dict(Counter(row["family"] for row in rows)),
        "core_decision_accuracy": mean(row["core_correct"] for row in rows),
        "horizon_decision_accuracy": mean(row["horizon_correct"] for row in rows),
        "designated_hidden_recall": mean(found),
        "designated_hidden_top3_recall": mean(1 if 0 < row["designated_hidden_rank"] <= 3 else 0 for row in rows),
        "mean_hidden_rank_when_found": mean(ranks),
        "mean_decision_changing_blind_spots": mean(row["decision_changing_blind_spots"] for row in rows),
        "mean_additional_risk_elevating_clauses": mean(row["additional_risk_elevating_clauses"] for row in rows),
        "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "warning": "This is a deterministic retrieval-blind-spot stress test, not an LLM quality result.",
        "rows": rows,
    }
    (output_dir / "horizon_challenge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic VeriWeave evidence-horizon challenge.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--challenge-file", default="data/tasks/horizon_challenge.jsonl")
    parser.add_argument("--output-dir", default="result/challenge")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    challenge = Path(args.challenge_file)
    if not challenge.is_absolute():
        challenge = root / challenge
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    summary = run_challenge(root, challenge, output)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
