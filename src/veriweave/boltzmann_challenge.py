from __future__ import annotations

import csv
import json
from pathlib import Path

from .boltzmann_policy_attention import attend_policy_candidates
from .vita import analyze_vita_decision_space
from .graph import Edge, Node, PropertyGraph
from .models import AtomicClaim
from .retrieval import Evidence


def _evidence(clause_id: str, citation: str, text: str, modality: str, concepts: list[str], *, current: bool, rank: int, score: float, source: str | None = None) -> Evidence:
    return Evidence(
        clause_id=clause_id,
        citation_id=citation,
        text=text,
        score=score,
        source=source or citation.split("#")[0],
        version="v2" if current else "v1",
        concepts=concepts,
        graph_path=[clause_id],
        role="challenge",
        retriever="boltzmann-challenge",
        modality=modality,
        is_current=current,
        authority_rank=rank,
    )


def _fixture() -> tuple[PropertyGraph, list[Evidence], list[Evidence], list[AtomicClaim]]:
    graph = PropertyGraph()
    for doc_id, version, rank in (("doc:seed", "v1", 1), ("doc:old", "v1", 1), ("doc:new", "v2", 3), ("doc:a", "v1", 2), ("doc:b", "v1", 2)):
        graph.add_node(Node(doc_id, "Document", doc_id, {"version": version, "authority_rank": rank}))
    graph.add_node(Node("concept:review", "Concept", "human-review", {}))
    graph.add_node(Node("concept:audit", "Concept", "auditability", {}))
    rows = [
        ("clause:seed", "The AI system may decide without human review.", "seed.md#rule@v1", "permission", True, 1, ["human-review"], "doc:seed"),
        ("clause:old", "The AI system may decide without human review during a pilot.", "policy.md#rule@v1", "permission", False, 1, ["human-review"], "doc:old"),
        ("clause:new", "The AI system must not decide without human review.", "policy.md#rule@v2", "prohibition", True, 3, ["human-review"], "doc:new"),
        ("clause:a", "Audit copies are retained for the pilot.", "a.md#audit@v1", "guidance", True, 2, ["auditability"], "doc:a"),
        ("clause:b", "Audit copies are deleted after the pilot.", "b.md#audit@v1", "guidance", True, 2, ["auditability"], "doc:b"),
    ]
    for clause_id, text, citation, modality, current, rank, concepts, doc_id in rows:
        graph.add_node(Node(clause_id, "Clause", clause_id, {
            "text": text,
            "citation_id": citation,
            "source": citation.split("#")[0],
            "version": "v2" if clause_id == "clause:new" else "v1",
            "concepts": concepts,
            "modality": modality,
            "is_current": current,
            "authority_rank": rank,
        }))
        graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
    for clause_id in ("clause:seed", "clause:old", "clause:new"):
        graph.add_edge(Edge(clause_id, "concept:review", "APPLIES_TO"))
    for clause_id in ("clause:a", "clause:b"):
        graph.add_edge(Edge(clause_id, "concept:audit", "APPLIES_TO"))
    graph.add_edge(Edge("clause:old", "clause:new", "CONTRADICTS"))
    graph.add_edge(Edge("clause:new", "clause:old", "CONTRADICTS"))
    graph.add_edge(Edge("doc:new", "doc:old", "OVERRIDES"))
    graph.add_edge(Edge("clause:a", "clause:b", "CONTRADICTS"))
    graph.add_edge(Edge("clause:b", "clause:a", "CONTRADICTS"))
    seed = [_evidence("clause:seed", "seed.md#rule@v1", rows[0][1], "permission", ["human-review"], current=True, rank=1, score=0.92)]
    candidates = [
        _evidence("clause:old", "policy.md#rule@v1", rows[1][1], "permission", ["human-review"], current=False, rank=1, score=0.57, source="policy.md"),
        _evidence("clause:new", "policy.md#rule@v2", rows[2][1], "prohibition", ["human-review"], current=True, rank=3, score=0.59, source="policy.md"),
        _evidence("clause:a", "a.md#audit@v1", rows[3][1], "guidance", ["auditability"], current=True, rank=2, score=0.61),
        _evidence("clause:b", "b.md#audit@v1", rows[4][1], "guidance", ["auditability"], current=True, rank=2, score=0.60),
    ]
    claims = [AtomicClaim("claim:decision", "The AI system is allowed to decide without human review.", "decision", "positive", True)]
    return graph, seed, candidates, claims


def main() -> None:
    graph, seed, candidates, claims = _fixture()
    selected, base = attend_policy_candidates(
        "Can the AI decide without human review?", claims, seed, candidates, graph, "allowed",
        base_temperature=0.75, max_coalition_size=2, selection_budget=3, attention_mass=0.90,
    )
    _, cold = attend_policy_candidates(
        "Can the AI decide without human review?", claims, seed, candidates, graph, "allowed",
        base_temperature=0.15, minimum_temperature=0.05, maximum_temperature=2.0,
        max_coalition_size=2, selection_budget=3,
    )
    _, hot = attend_policy_candidates(
        "Can the AI decide without human review?", claims, seed, candidates, graph, "allowed",
        base_temperature=1.80, minimum_temperature=0.05, maximum_temperature=3.0,
        max_coalition_size=2, selection_budget=3,
    )
    vita, *_middle, integrated = analyze_vita_decision_space(
        "Can the AI decide without human review?", claims, seed, graph, "allowed",
        max_candidates=4, max_coalition_size=2, max_rounds=1,
        use_boltzmann_attention=True, boltzmann_max_candidates=4,
    )
    probability_sum = sum(row["probability"] for row in base.top_coalitions)
    cases = [
        {
            "case": "normalized_policy_set_distribution",
            "success": abs(probability_sum - 1.0) < 1e-5,
            "probability_sum": probability_sum,
        },
        {
            "case": "temperature_controls_exploration",
            "success": cold.normalized_entropy < hot.normalized_entropy,
            "cold_entropy": cold.normalized_entropy,
            "hot_entropy": hot.normalized_entropy,
        },
        {
            "case": "coupling_surfaces_current_counterrule",
            "success": "clause:new" in base.selected_clause_ids and any(
                row["coupling"] > 0.2 and {row["left_clause_id"], row["right_clause_id"]} == {"clause:old", "clause:new"}
                for row in base.pairwise_couplings
            ),
            "selected_clause_ids": base.selected_clause_ids,
        },
        {
            "case": "vita_integration_emits_certificate",
            "success": integrated is not None and "needs_review" in vita.reachable_decisions,
            "reachable_decisions": vita.reachable_decisions,
            "selected_clause_ids": integrated.selected_clause_ids if integrated else [],
        },
    ]
    summary = {
        "challenge": "VeriWeave-VITA-BPA deterministic mechanism challenge",
        "warning": "These checks validate deterministic policy-attention mechanics, not LLM answer quality.",
        "cases": cases,
        "passed": sum(1 for item in cases if item["success"]),
        "total": len(cases),
        "certificate": base.to_dict(),
    }
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "result" / "challenge"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "boltzmann_policy_attention_challenge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "boltzmann_policy_attention_challenge.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "success"])
        writer.writeheader()
        writer.writerows({"case": item["case"], "success": item["success"]} for item in cases)
    print(json.dumps({"passed": summary["passed"], "total": summary["total"]}, indent=2))


if __name__ == "__main__":
    main()
