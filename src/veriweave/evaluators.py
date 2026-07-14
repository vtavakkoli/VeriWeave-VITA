from __future__ import annotations

from .data_loader import BenchmarkTask
from .utils import mean, token_similarity


def _compatible_decision(expected: str, actual: str) -> float:
    if expected == actual:
        return 1.0
    compatible = {
        ("conditional", "needs_review"),
        ("needs_review", "conditional"),
        ("not_allowed", "needs_review"),
        ("allowed", "conditional"),
    }
    return 0.5 if (expected, actual) in compatible else 0.0


def evaluate(task: BenchmarkTask, trace: dict) -> dict:
    validations = trace.get("audit_validations", [])
    claims = trace.get("audit_claims", [])
    audit_evidence = trace.get("audit_evidence", [])
    audit_horizon = trace.get("audit_horizon", {}) or {}
    audit_vita = trace.get("audit_vita", {}) or {}
    audit_argumentation = trace.get("audit_argumentation", {}) or {}
    audit_temporal = trace.get("audit_temporal_drift", {}) or {}
    audit_attention = trace.get("audit_boltzmann_policy_attention", {}) or {}
    method_evidence = trace.get("evidence", [])
    method_horizon = trace.get("evidence_horizon_certificate", {}) or {}
    method_vita = trace.get("vita_certificate", {}) or {}
    method_attention = trace.get("boltzmann_policy_attention_certificate", {}) or {}
    citations = trace.get("citations", [])

    supported = sum(1 for item in validations if item.get("status") == "supported")
    contradicted = sum(1 for item in validations if item.get("status") == "contradicted")
    unresolved = sum(1 for item in validations if item.get("status") == "unresolved")
    claim_precision = supported / max(1, len(validations))
    unsupported_claim_rate = (contradicted + unresolved) / max(1, len(validations))
    contradiction_safety = 1.0 if contradicted == 0 or trace.get("human_review_required") else 0.0
    provenance_coverage = sum(1 for item in validations if item.get("provenance_complete")) / max(1, len(validations))
    temporal_validity = mean(float(item.get("temporal_validity_score", 0.0)) for item in validations)
    applicability = mean(float(item.get("applicability_score", 0.0)) for item in validations)

    audit_citations = {str(item.get("citation_id", "")) for item in audit_evidence}
    valid_citations = [citation for citation in citations if citation in audit_citations]
    citation_precision = len(valid_citations) / max(1, len(citations)) if citations else 0.0
    supported_evidence_ids = {
        citation
        for item in validations
        if item.get("status") == "supported"
        for citation in item.get("evidence_ids", [])
    }
    citation_coverage = len(set(citations) & supported_evidence_ids) / max(1, len(supported_evidence_ids)) if supported_evidence_ids else 0.0

    expected_refs = set(task.expected_policy_refs)
    method_evidence_text = " ".join(
        f"{item.get('citation_id', '')} {item.get('text', '')}" for item in method_evidence
    ).lower()
    graph_reference_recall = sum(1 for ref in expected_refs if ref.lower() in method_evidence_text) / max(1, len(expected_refs))

    answer_similarity = token_similarity(str(trace.get("answer", "")), task.expected_answer)
    actual_decision = str(trace.get("decision", "unknown"))
    decision_accuracy = 1.0 if task.expected_decision == actual_decision else 0.0
    decision_compatibility = _compatible_decision(task.expected_decision, actual_decision)
    review_expected = task.expected_human_review
    review_accuracy = 1.0 if review_expected is None or bool(review_expected) == bool(trace.get("human_review_required")) else 0.0

    graph_paths = sum(1 for item in method_evidence if item.get("graph_path")) / max(1, len(method_evidence)) if method_evidence else 0.0
    has_envelope = 1.0 if trace.get("verification_envelope") else 0.0
    has_resolution = 1.0 if trace.get("method_resolutions") else 0.0
    traceability = mean([citation_precision, citation_coverage, graph_paths, max(has_envelope, has_resolution)])

    counterfactual_stability = float(audit_horizon.get("horizon_stability", 0.0))
    evidence_cut_robustness = float(audit_horizon.get("evidence_cut_robustness", 0.0))
    decision_changing_blind_spots = len(audit_horizon.get("decision_changing_blind_spots", []))
    unseen_evidence_rate = float(audit_horizon.get("unseen_evidence_rate", 0.0))
    decision_space_width = int(audit_vita.get("decision_space_width", 0))
    vita_invariance = 1.0 if audit_vita.get("invariant_under_tested_closure", False) else 0.0
    closure_convergence = 1.0 if audit_vita.get("closure_converged", False) else 0.0
    residual_risk_mass = float(audit_vita.get("residual_risk_mass", 1.0))
    synergistic_blind_spots = len(audit_vita.get("synergistic_blind_spots", []))
    vita_tested_subsets = int(audit_vita.get("tested_subsets", 0))
    argumentation_conflict_free = 1.0 if audit_argumentation.get("conflict_free", False) else 0.0
    temporal_stability = 1.0 if audit_temporal.get("stable_across_versions", False) else 0.0
    attention_selected_mass = float(audit_attention.get("selected_attention_mass", 0.0))
    attention_covered_coalition_probability = float(audit_attention.get("covered_coalition_probability", 0.0))
    attention_residual_mass = float(audit_attention.get("residual_probability_mass", 1.0))
    attention_entropy = float(audit_attention.get("normalized_entropy", 0.0))
    attention_effective_sample_size = float(audit_attention.get("effective_sample_size", 0.0))
    attention_risk_tail_mass = float(audit_attention.get("risk_tail_mass", 0.0))
    # Robustness receives credit only when the answer contains independently supported claims.
    robustness_quality = mean([
        counterfactual_stability,
        evidence_cut_robustness,
        vita_invariance,
        closure_convergence * (1.0 - residual_risk_mass),
        argumentation_conflict_free,
        temporal_stability,
        attention_selected_mass,
    ]) * claim_precision

    method_blind_spots_found = len(method_horizon.get("blind_spots", []))
    method_decision_changing_found = len(method_horizon.get("decision_changing_blind_spots", []))
    method_horizon_expansion = len(method_horizon.get("expanded_evidence_ids", []))
    method_coalitional_blind_spots = len(method_vita.get("coalitional_blind_spots", []))
    method_synergistic_blind_spots = len(method_vita.get("synergistic_blind_spots", []))
    method_vita_width = int(method_vita.get("decision_space_width", 0))
    method_vita_tested_subsets = int(method_vita.get("tested_subsets", 0))
    method_attention_selected_mass = float(method_attention.get("selected_attention_mass", 0.0))
    method_attention_covered_coalition_probability = float(method_attention.get("covered_coalition_probability", 0.0))
    method_attention_residual_mass = float(method_attention.get("residual_probability_mass", 1.0))
    method_attention_entropy = float(method_attention.get("normalized_entropy", 0.0))
    method_attention_effective_sample_size = float(method_attention.get("effective_sample_size", 0.0))
    method_attention_risk_tail_mass = float(method_attention.get("risk_tail_mass", 0.0))
    method_attention_selected_clauses = len(method_attention.get("selected_clause_ids", []))

    answer_quality = mean([decision_accuracy, review_accuracy, answer_similarity])
    verification_quality = mean([claim_precision, 1.0 - unsupported_claim_rate, contradiction_safety])
    provenance_quality = mean([provenance_coverage, temporal_validity, applicability, citation_precision, citation_coverage])
    retrieval_quality = graph_reference_recall
    overall = (
        0.35 * answer_quality
        + 0.25 * verification_quality
        + 0.15 * provenance_quality
        + 0.10 * retrieval_quality
        + 0.15 * robustness_quality
    )

    return {
        "answer_similarity": round(answer_similarity, 6),
        "decision_accuracy": round(decision_accuracy, 6),
        "decision_compatibility": round(decision_compatibility, 6),
        "review_routing_accuracy": round(review_accuracy, 6),
        "claim_precision": round(claim_precision, 6),
        "unsupported_claim_rate": round(unsupported_claim_rate, 6),
        "contradiction_safety": round(contradiction_safety, 6),
        "provenance_coverage": round(provenance_coverage, 6),
        "temporal_validity_accuracy": round(temporal_validity, 6),
        "applicability_accuracy": round(applicability, 6),
        "citation_precision": round(citation_precision, 6),
        "citation_coverage": round(citation_coverage, 6),
        "graph_reference_recall": round(graph_reference_recall, 6),
        "traceability_score": round(traceability, 6),
        "counterfactual_stability": round(counterfactual_stability, 6),
        "evidence_cut_robustness": round(evidence_cut_robustness, 6),
        "decision_changing_blind_spots": decision_changing_blind_spots,
        "unseen_evidence_rate": round(unseen_evidence_rate, 6),
        "decision_space_width": decision_space_width,
        "vita_invariance": round(vita_invariance, 6),
        "closure_convergence": round(closure_convergence, 6),
        "residual_risk_mass": round(residual_risk_mass, 6),
        "synergistic_blind_spots": synergistic_blind_spots,
        "vita_tested_subsets": vita_tested_subsets,
        "argumentation_conflict_free": round(argumentation_conflict_free, 6),
        "temporal_stability": round(temporal_stability, 6),
        "attention_selected_mass": round(attention_selected_mass, 6),
        "attention_covered_coalition_probability": round(attention_covered_coalition_probability, 6),
        "attention_residual_mass": round(attention_residual_mass, 6),
        "attention_entropy": round(attention_entropy, 6),
        "attention_effective_sample_size": round(attention_effective_sample_size, 6),
        "attention_risk_tail_mass": round(attention_risk_tail_mass, 6),
        "robustness_quality": round(robustness_quality, 6),
        "method_blind_spots_found": method_blind_spots_found,
        "method_decision_changing_found": method_decision_changing_found,
        "method_horizon_expansion": method_horizon_expansion,
        "method_coalitional_blind_spots": method_coalitional_blind_spots,
        "method_synergistic_blind_spots": method_synergistic_blind_spots,
        "method_vita_width": method_vita_width,
        "method_vita_tested_subsets": method_vita_tested_subsets,
        "method_attention_selected_mass": round(method_attention_selected_mass, 6),
        "method_attention_covered_coalition_probability": round(method_attention_covered_coalition_probability, 6),
        "method_attention_residual_mass": round(method_attention_residual_mass, 6),
        "method_attention_entropy": round(method_attention_entropy, 6),
        "method_attention_effective_sample_size": round(method_attention_effective_sample_size, 6),
        "method_attention_risk_tail_mass": round(method_attention_risk_tail_mass, 6),
        "method_attention_selected_clauses": method_attention_selected_clauses,
        "answer_quality": round(answer_quality, 6),
        "verification_quality": round(verification_quality, 6),
        "provenance_quality": round(provenance_quality, 6),
        "overall_score": round(overall, 6),
        "task_success": 1.0 if decision_accuracy == 1.0 and unsupported_claim_rate <= 0.5 and overall >= 0.60 else 0.0,
        "claim_count": len(claims),
        "method_evidence_count": len(method_evidence),
        "audit_evidence_count": len(audit_evidence),
        "latency_seconds": float(trace.get("latency_seconds", 0.0)),
    }
