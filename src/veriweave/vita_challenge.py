from __future__ import annotations

"""Deterministic mechanism challenge for VeriWeave-VITA.

This is not an LLM benchmark. It isolates three mechanisms: coalitional blind
spots, precedence-grounded argumentation, and temporal decision drift.
"""

import csv
import json
from pathlib import Path

from .argumentation import build_argumentation_certificate
from .vita import analyze_vita_decision_space
from .graph import Edge, Node, PropertyGraph
from .models import AtomicClaim
from .precedence import resolve_precedence
from .retrieval import Evidence
from .temporal import build_temporal_drift_certificate


def _evidence(clause_id: str, citation: str, text: str, modality: str, concepts: list[str], *, current: bool, rank: int, score: float = 0.8) -> Evidence:
    return Evidence(
        clause_id=clause_id,
        citation_id=citation,
        text=text,
        score=score,
        source=citation.split("#")[0],
        version="v2" if current else "v1",
        concepts=concepts,
        graph_path=[clause_id],
        role="challenge",
        retriever="vita-challenge",
        modality=modality,
        is_current=current,
        authority_rank=rank,
    )


def _coalition_case() -> dict:
    graph = PropertyGraph()
    for doc_id, label in (("doc:base", "Base Policy"), ("doc:a", "Audit A"), ("doc:b", "Audit B")):
        graph.add_node(Node(doc_id, "Document", label, {"version": "v1", "authority_rank": 2}))
    graph.add_node(Node("concept:human-review", "Concept", "human-review", {}))
    graph.add_node(Node("concept:auditability", "Concept", "auditability", {}))
    clauses = [
        ("clause:base", "Decision", "The AI system may make the decision without human review.", "base.md#decision@v1", ["human-review", "auditability"], "permission"),
        ("clause:a", "Audit retention", "Audit copies are retained for the pilot.", "a.md#audit@v1", ["auditability"], "guidance"),
        ("clause:b", "Audit deletion", "Audit copies are deleted after the pilot.", "b.md#audit@v1", ["auditability"], "guidance"),
    ]
    for clause_id, label, text, citation, concepts, modality in clauses:
        graph.add_node(Node(clause_id, "Clause", label, {
            "text": text, "citation_id": citation, "source": citation.split("#")[0],
            "version": "v1", "concepts": concepts, "modality": modality,
            "is_current": True, "authority_rank": 2,
        }))
    for clause_id, doc_id in (("clause:base", "doc:base"), ("clause:a", "doc:a"), ("clause:b", "doc:b")):
        graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
    graph.add_edge(Edge("clause:base", "concept:human-review", "APPLIES_TO"))
    graph.add_edge(Edge("clause:base", "concept:auditability", "APPLIES_TO"))
    graph.add_edge(Edge("clause:a", "concept:auditability", "APPLIES_TO"))
    graph.add_edge(Edge("clause:b", "concept:auditability", "APPLIES_TO"))
    graph.add_edge(Edge("clause:a", "clause:b", "CONTRADICTS"))
    graph.add_edge(Edge("clause:b", "clause:a", "CONTRADICTS"))
    initial = [_evidence(
        "clause:base", "base.md#decision@v1",
        "The AI system may make the decision without human review.",
        "permission", ["human-review", "auditability"], current=True, rank=2, score=0.95,
    )]
    claims = [AtomicClaim(
        id="claim:decision",
        text="The AI system is allowed to make the decision without human review.",
        kind="decision", decisive=True,
    )]
    certificate, *_ = analyze_vita_decision_space(
        "Can the AI decide without human review?", claims, initial, graph, "allowed",
        max_candidates=4, max_coalition_size=2, max_rounds=1,
    )
    success = bool(certificate.synergistic_blind_spots) and "needs_review" in certificate.reachable_decisions
    return {
        "case": "coalitional_blind_spot",
        "success": success,
        "singleton_missed_joint_conflict": bool(certificate.synergistic_blind_spots),
        "reachable_decisions": certificate.reachable_decisions,
        "vita_width": certificate.decision_space_width,
        "certificate": certificate.to_dict(),
    }



def _two_sided_case() -> dict:
    graph = PropertyGraph()
    graph.add_node(Node("doc:scope", "Document", "Cloud Scope", {"version": "v1", "authority_rank": 2}))
    graph.add_node(Node("doc:permission", "Document", "Cloud Permission", {"version": "v1", "authority_rank": 2}))
    graph.add_node(Node("concept:external-service", "Concept", "external-service", {}))
    graph.add_node(Node("clause:scope", "Clause", "Scope", {
        "text": "This policy governs the use of external cloud services.",
        "citation_id": "scope.md#scope@v1", "source": "scope.md", "version": "v1",
        "concepts": ["external-service"], "modality": "guidance", "is_current": True, "authority_rank": 2,
    }))
    graph.add_node(Node("clause:permission", "Clause", "Permission", {
        "text": "External cloud services may be used for approved pilot workloads.",
        "citation_id": "permission.md#cloud@v1", "source": "permission.md", "version": "v1",
        "concepts": ["external-service"], "modality": "permission", "is_current": True, "authority_rank": 2,
    }))
    graph.add_edge(Edge("clause:scope", "doc:scope", "DERIVED_FROM"))
    graph.add_edge(Edge("clause:permission", "doc:permission", "DERIVED_FROM"))
    graph.add_edge(Edge("clause:scope", "concept:external-service", "APPLIES_TO"))
    graph.add_edge(Edge("clause:permission", "concept:external-service", "APPLIES_TO"))
    initial = [_evidence(
        "clause:scope", "scope.md#scope@v1",
        "This policy governs the use of external cloud services.",
        "guidance", ["external-service"], current=True, rank=2, score=0.95,
    )]
    claims = [AtomicClaim(
        id="claim:scope", text="The policy addresses external cloud services.", kind="assertion", decisive=False,
    )]
    certificate, *_ = analyze_vita_decision_space(
        "Does the policy address external cloud services?", claims, initial, graph, "not_allowed",
        max_candidates=3, max_coalition_size=1, max_rounds=1,
    )
    success = certificate.most_permissive_decision == "allowed" and certificate.most_restrictive_decision == "not_allowed"
    return {
        "case": "two_sided_vita_certificate",
        "success": success,
        "most_permissive": certificate.most_permissive_decision,
        "most_restrictive": certificate.most_restrictive_decision,
        "vita_width": certificate.decision_space_width,
        "certificate": certificate.to_dict(),
    }

def _version_graph() -> tuple[PropertyGraph, list[Evidence], list[AtomicClaim]]:
    graph = PropertyGraph()
    graph.add_node(Node("doc:old", "Document", "Automated Decision Policy v1", {"version": "v1", "authority_rank": 1}))
    graph.add_node(Node("doc:new", "Document", "Automated Decision Policy v2", {"version": "v2", "authority_rank": 2}))
    graph.add_node(Node("clause:old", "Clause", "Human review", {
        "text": "The AI system may decide without human review.",
        "citation_id": "old.md#review@v1", "source": "old.md", "version": "v1",
        "concepts": ["human-review"], "modality": "permission", "is_current": False, "authority_rank": 1,
    }))
    graph.add_node(Node("clause:new", "Clause", "Human review", {
        "text": "The AI system must not decide without human review.",
        "citation_id": "new.md#review@v2", "source": "new.md", "version": "v2",
        "concepts": ["human-review"], "modality": "prohibition", "is_current": True, "authority_rank": 2,
    }))
    graph.add_edge(Edge("clause:old", "doc:old", "DERIVED_FROM"))
    graph.add_edge(Edge("clause:new", "doc:new", "DERIVED_FROM"))
    graph.add_edge(Edge("doc:new", "doc:old", "OVERRIDES"))
    graph.add_edge(Edge("clause:old", "clause:new", "CONTRADICTS"))
    graph.add_edge(Edge("clause:new", "clause:old", "CONTRADICTS"))
    old = _evidence("clause:old", "old.md#review@v1", "The AI system may decide without human review.", "permission", ["human-review"], current=False, rank=1)
    new = _evidence("clause:new", "new.md#review@v2", "The AI system must not decide without human review.", "prohibition", ["human-review"], current=True, rank=2)
    claims = [AtomicClaim(
        id="claim:decision",
        text="The AI system is allowed to decide without human review.",
        kind="decision", decisive=True,
    )]
    return graph, [old, new], claims


def _argumentation_case() -> dict:
    graph, evidence, _claims = _version_graph()
    resolutions = resolve_precedence(evidence, graph)
    certificate = build_argumentation_certificate(evidence, graph, resolutions)
    success = "clause:new" in certificate.accepted_argument_ids and "clause:old" in certificate.rejected_argument_ids
    return {
        "case": "precedence_grounded_argumentation",
        "success": success,
        "accepted": certificate.accepted_argument_ids,
        "rejected": certificate.rejected_argument_ids,
        "conflict_free": certificate.conflict_free,
        "certificate": certificate.to_dict(),
    }


def _temporal_case() -> dict:
    graph, _evidence_items, claims = _version_graph()
    certificate = build_temporal_drift_certificate(
        "Can the AI decide without human review?", claims, graph, "allowed", ["clause:old", "clause:new"]
    )
    success = certificate.retroactive_review_required and certificate.earliest_decision != certificate.latest_decision
    return {
        "case": "temporal_decision_drift",
        "success": success,
        "earliest_decision": certificate.earliest_decision,
        "latest_decision": certificate.latest_decision,
        "drift_events": len(certificate.drift_events),
        "certificate": certificate.to_dict(),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "result" / "challenge"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [_coalition_case(), _two_sided_case(), _argumentation_case(), _temporal_case()]
    summary = {
        "challenge": "VeriWeave-VITA deterministic mechanism challenge",
        "warning": "These are deterministic mechanism checks, not LLM-quality results.",
        "cases": cases,
        "passed": sum(1 for item in cases if item["success"]),
        "total": len(cases),
    }
    (out_dir / "vita_challenge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "vita_challenge.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "success"])
        writer.writeheader()
        writer.writerows({"case": item["case"], "success": item["success"]} for item in cases)
    print(json.dumps({"passed": summary["passed"], "total": summary["total"]}, indent=2))


if __name__ == "__main__":
    main()
