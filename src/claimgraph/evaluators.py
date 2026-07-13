from __future__ import annotations

from .data_loader import BenchmarkTask
from .utils import mean, token_similarity, tokens


def _decision_accuracy(expected: str, actual: str) -> float:
    if expected == actual:
        return 1.0
    compatible = {
        ("conditional", "needs_review"), ("needs_review", "conditional"),
        ("not_allowed", "needs_review"), ("allowed", "conditional"),
    }
    return 0.5 if (expected, actual) in compatible else 0.0


def evaluate(task: BenchmarkTask, trace: dict) -> dict:
    validations = trace.get("validations", [])
    claims = trace.get("claims", [])
    evidence = trace.get("evidence", [])
    citations = trace.get("citations", [])
    supported = sum(1 for item in validations if item.get("status") == "supported")
    contradicted = sum(1 for item in validations if item.get("status") == "contradicted")
    unresolved = max(0, len(claims) - supported - contradicted)

    claim_precision = supported / max(1, len(claims)) if validations else 0.0
    unsupported_claim_rate = (unresolved + contradicted) / max(1, len(claims)) if claims else 1.0
    contradiction_detection = 1.0 if contradicted and trace.get("human_review_required") else (0.0 if contradicted else 1.0)
    provenance_coverage = sum(1 for item in validations if item.get("provenance_complete")) / max(1, len(validations)) if validations else 0.0
    temporal_validity = mean(float(item.get("temporal_validity_score", 0.0)) for item in validations) if validations else 0.0
    applicability = mean(float(item.get("applicability_score", 0.0)) for item in validations) if validations else 0.0
    citation_precision = sum(1 for citation in citations if any(citation == item.get("citation_id") for item in evidence)) / max(1, len(citations)) if citations else 0.0
    expected_refs = set(task.expected_policy_refs)
    evidence_text = " ".join([str(item.get("citation_id", "")) + " " + str(item.get("text", "")) for item in evidence]).lower()
    graph_ref_recall = sum(1 for ref in expected_refs if ref.lower() in evidence_text) / max(1, len(expected_refs))
    answer_similarity = token_similarity(trace.get("answer", ""), task.expected_answer)
    decision_accuracy = _decision_accuracy(task.expected_decision, trace.get("decision", "unknown"))
    review_expected = task.expected_human_review
    review_accuracy = 1.0 if review_expected is None or bool(review_expected) == bool(trace.get("human_review_required")) else 0.0
    explanation_nodes = len(trace.get("claim_evidence_subgraph", {}).get("nodes", []))
    traceability = min(1.0, (0.5 if evidence else 0.0) + (0.3 if citations else 0.0) + (0.2 if explanation_nodes else 0.0))

    reliability = mean([
        decision_accuracy,
        review_accuracy,
        graph_ref_recall,
        contradiction_detection,
        max(claim_precision, answer_similarity * 0.5),
    ])
    trustworthiness = mean([
        provenance_coverage,
        temporal_validity,
        applicability,
        traceability,
        1.0 - unsupported_claim_rate,
    ])
    overall = 0.55 * reliability + 0.45 * trustworthiness

    return {
        "answer_similarity": round(answer_similarity, 6),
        "decision_accuracy": round(decision_accuracy, 6),
        "review_routing_accuracy": round(review_accuracy, 6),
        "claim_precision": round(claim_precision, 6),
        "unsupported_claim_rate": round(unsupported_claim_rate, 6),
        "contradiction_detection": round(contradiction_detection, 6),
        "provenance_coverage": round(provenance_coverage, 6),
        "temporal_validity_accuracy": round(temporal_validity, 6),
        "applicability_accuracy": round(applicability, 6),
        "citation_precision": round(citation_precision, 6),
        "graph_reference_recall": round(graph_ref_recall, 6),
        "traceability_score": round(traceability, 6),
        "reliability_score": round(reliability, 6),
        "trustworthiness_score": round(trustworthiness, 6),
        "overall_score": round(overall, 6),
        "task_success": 1.0 if overall >= 0.60 and decision_accuracy >= 0.5 else 0.0,
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "latency_seconds": float(trace.get("latency_seconds", 0.0)),
    }
