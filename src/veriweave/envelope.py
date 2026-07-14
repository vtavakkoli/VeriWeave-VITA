from __future__ import annotations

from .claims import extract_atomic_claims
from .vita import analyze_vita_decision_space
from .graph import PropertyGraph
from .horizon import analyze_evidence_horizon
from .models import CandidateResponse, VerificationEnvelope
from .precedence import resolve_precedence
from .retrieval import Evidence
from .utils import mean, stable_hash
from typing import Any
from .validators import validate_claims


def build_verification_envelope(
    candidate: CandidateResponse,
    evidence: list[Evidence],
    graph: PropertyGraph,
    *,
    question: str = "",
    enable_horizon: bool = True,
    enable_vita: bool = False,
    enable_boltzmann_attention: bool = False,
    vita_options: dict[str, Any] | None = None,
) -> VerificationEnvelope:
    claims = extract_atomic_claims(candidate.answer)
    horizon_certificate = None
    vita_certificate = None
    argumentation_certificate = None
    temporal_certificate = None
    boltzmann_certificate = None

    if enable_vita:
        (
            vita_certificate,
            effective_evidence,
            resolutions,
            validations,
            argumentation_certificate,
            temporal_certificate,
            boltzmann_certificate,
        ) = analyze_vita_decision_space(
            question or candidate.answer,
            claims,
            evidence,
            graph,
            candidate.decision,
            use_boltzmann_attention=enable_boltzmann_attention,
            **(vita_options or {}),
        )
    elif enable_horizon:
        horizon_certificate, effective_evidence, resolutions, validations = analyze_evidence_horizon(
            question or candidate.answer,
            claims,
            evidence,
            graph,
            candidate.decision,
        )
    else:
        effective_evidence = evidence
        resolutions = resolve_precedence(effective_evidence, graph)
        validations = validate_claims(claims, effective_evidence, graph, resolutions)

    validation_by_id = {item.claim_id: item for item in validations}

    retained_claims = []
    removed_claim_ids = []
    for claim in claims:
        validation = validation_by_id[claim.id]
        if validation.status == "supported":
            retained_claims.append(claim.text)
        else:
            removed_claim_ids.append(claim.id)

    review_reasons: list[str] = []
    decisive_failures = [
        claim for claim in claims
        if claim.decisive and validation_by_id[claim.id].status != "supported"
    ]
    if decisive_failures:
        review_reasons.append("one or more decision-bearing claims were not verified")
    if any(item.status == "contradicted" for item in validations):
        review_reasons.append("counterevidence contradicts at least one candidate claim")
    if any(record.unresolved for record in resolutions):
        review_reasons.append("policy precedence could not resolve all conflicts")
    if any(not item.provenance_complete for item in validations):
        review_reasons.append("at least one claim lacks complete provenance")
    if removed_claim_ids:
        review_reasons.append("unsupported claims were removed by the claim-selective gate")
    if horizon_certificate:
        for reason in horizon_certificate.review_reasons:
            if reason not in review_reasons:
                review_reasons.append(reason)
    if vita_certificate:
        for reason in vita_certificate.review_reasons:
            if reason not in review_reasons:
                review_reasons.append(reason)
    if argumentation_certificate and argumentation_certificate.undecided_argument_ids:
        review_reasons.append("grounded argumentation leaves one or more evidence arguments undecided")
    if temporal_certificate and temporal_certificate.retroactive_review_required:
        review_reasons.append("policy-version replay detects a more restrictive decision drift")
    if boltzmann_certificate and boltzmann_certificate.residual_probability_mass > 0.35:
        review_reasons.append("Boltzmann policy attention leaves material probability mass outside the selected clause budget")

    accepted_citations = sorted({
        citation
        for validation in validations
        if validation.status == "supported"
        for citation in validation.evidence_ids
        if citation in candidate.citations
    })
    evidence_citations = {item.citation_id for item in effective_evidence}
    rejected_citations = sorted({citation for citation in candidate.citations if citation not in evidence_citations})

    if retained_claims:
        verified_answer = " ".join(retained_claims)
    else:
        verified_answer = "The available evidence is insufficient to verify the requested conclusion."
    if review_reasons:
        verified_answer = f"{verified_answer} Human review is required."

    supported_ratio = sum(1 for item in validations if item.status == "supported") / max(1, len(validations))
    provenance_ratio = sum(1 for item in validations if item.provenance_complete) / max(1, len(validations))
    resolution_ratio = sum(1 for item in resolutions if not item.unresolved) / max(1, len(resolutions)) if resolutions else 1.0
    horizon_stability = horizon_certificate.horizon_stability if horizon_certificate else 1.0
    cut_robustness = horizon_certificate.evidence_cut_robustness if horizon_certificate else 1.0
    vita_invariance = (
        1.0 if vita_certificate and vita_certificate.invariant_under_tested_closure
        else 0.0 if vita_certificate else 1.0
    )
    closure_quality = (
        (1.0 if vita_certificate.closure_converged else 0.0) * (1.0 - vita_certificate.residual_risk_mass)
        if vita_certificate else 1.0
    )
    argumentation_quality = (
        1.0 if argumentation_certificate and argumentation_certificate.conflict_free else 0.0
        if argumentation_certificate else 1.0
    )
    temporal_stability = (
        1.0 if temporal_certificate and temporal_certificate.stable_across_versions else 0.0
        if temporal_certificate else 1.0
    )
    attention_coverage = boltzmann_certificate.selected_attention_mass if boltzmann_certificate else 1.0
    confidence = mean([
        supported_ratio,
        provenance_ratio,
        resolution_ratio,
        horizon_stability,
        cut_robustness,
        vita_invariance,
        closure_quality,
        argumentation_quality,
        temporal_stability,
        attention_coverage,
    ])

    gated_decision = candidate.decision
    human_review_required = candidate.human_review_required
    if review_reasons or candidate.decision == "unknown":
        gated_decision = "needs_review"
        human_review_required = True

    graph_digest = stable_hash(
        "|".join(sorted(item.clause_id for item in effective_evidence))
        + "|"
        + "|".join(sorted(record.id for record in resolutions)),
        16,
    )
    envelope_id = f"envelope:{stable_hash(candidate.answer + graph_digest)}"

    return VerificationEnvelope(
        envelope_id=envelope_id,
        candidate_answer=candidate.answer,
        verified_answer=verified_answer,
        candidate_decision=candidate.decision,
        gated_decision=gated_decision,
        candidate_review_required=candidate.human_review_required,
        human_review_required=human_review_required,
        confidence=round(confidence, 6),
        claims=[claim.to_dict() for claim in claims],
        validations=[item.to_dict() for item in validations],
        resolutions=[record.to_dict() for record in resolutions],
        accepted_citations=accepted_citations,
        rejected_citations=rejected_citations,
        removed_claim_ids=removed_claim_ids,
        review_reasons=review_reasons,
        graph_digest=graph_digest,
        evidence_horizon=horizon_certificate.to_dict() if horizon_certificate else None,
        vita_certificate=vita_certificate.to_dict() if vita_certificate else None,
        argumentation_certificate=argumentation_certificate.to_dict() if argumentation_certificate else None,
        temporal_drift_certificate=temporal_certificate.to_dict() if temporal_certificate else None,
        boltzmann_policy_attention=boltzmann_certificate.to_dict() if boltzmann_certificate else None,
    )
