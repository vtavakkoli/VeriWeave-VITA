from __future__ import annotations

from .claims import extract_atomic_claims
from .vita import analyze_vita_decision_space
from .graph import PropertyGraph
from .horizon import analyze_evidence_horizon
from .precedence import resolve_precedence
from .retrieval import AuditRetriever
from .validators import validate_claims


def attach_independent_audit(trace: dict, graph: PropertyGraph, retriever: AuditRetriever, top_k: int = 10) -> dict:
    """Attach identical post-hoc verification and robustness audits to every method."""
    claims = extract_atomic_claims(str(trace.get("answer", "")))
    evidence = retriever.retrieve(str(trace.get("question", "")), str(trace.get("answer", "")), top_k)
    resolutions = resolve_precedence(evidence, graph)
    validations = validate_claims(claims, evidence, graph, resolutions)
    horizon, _, _, _ = analyze_evidence_horizon(
        str(trace.get("question", "")),
        claims,
        evidence,
        graph,
        str(trace.get("decision", "unknown")),
        max_candidates=10,
        max_expand=3,
        max_cut_size=3,
    )
    vita, _, _, _, argumentation, temporal, boltzmann_attention = analyze_vita_decision_space(
        str(trace.get("question", "")),
        claims,
        evidence,
        graph,
        str(trace.get("decision", "unknown")),
        max_candidates=6,
        max_coalition_size=2,
        max_rounds=1,
        max_expand_per_round=3,
        use_boltzmann_attention=True,
        boltzmann_max_candidates=10,
        boltzmann_temperature=0.75,
        boltzmann_attention_mass=0.90,
    )
    trace["audit_claims"] = [claim.to_dict() for claim in claims]
    trace["audit_evidence"] = [item.to_dict() for item in evidence]
    trace["audit_resolutions"] = [record.to_dict() for record in resolutions]
    trace["audit_validations"] = [item.to_dict() for item in validations]
    trace["audit_horizon"] = horizon.to_dict()
    trace["audit_vita"] = vita.to_dict()
    trace["audit_argumentation"] = argumentation.to_dict()
    trace["audit_temporal_drift"] = temporal.to_dict()
    trace["audit_boltzmann_policy_attention"] = boltzmann_attention.to_dict() if boltzmann_attention else None
    return trace
