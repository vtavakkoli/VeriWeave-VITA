from __future__ import annotations

import re
from typing import Any

from .claims import extract_atomic_claims
from .graph import PropertyGraph
from .horizon import analyze_evidence_horizon
from .models import CandidateResponse, VALID_DECISIONS, VerificationEnvelope
from .precedence import resolve_precedence
from .retrieval import Evidence
from .utils import mean, stable_hash, strip_citations
from .validators import validate_claims
from .vita import analyze_vita_decision_space

_REVIEW_PATTERNS = (
    r"human (?:review|oversight) (?:is )?(?:required|mandatory)",
    r"requires? (?:a )?(?:trained )?human (?:review|oversight)",
    r"must (?:be )?reviewed by (?:a )?human",
    r"must not .* without (?:a )?(?:trained )?human (?:review|oversight)",
    r"(?:review|approval|oversight) (?:is )?required before",
    r"must (?:be )?approved before",
    r"route(?:d)? (?:the case )?for review",
)
_CONFLICT_CUES = (
    "conflict",
    "contradict",
    "which applies",
    "older",
    "newer",
    "current version",
    "supersed",
    "override",
)


def _requires_human_review(text: str) -> bool:
    lower = f" {strip_citations(text).lower()} "
    if "human-review flag" in lower or "human review flag" in lower:
        return False
    return any(re.search(pattern, lower) for pattern in _REVIEW_PATTERNS)


def _append_unique(target: list[str], reason: str) -> None:
    if reason and reason not in target:
        target.append(reason)


def _relevant_clause_ids(
    claims,
    validations,
    evidence: list[Evidence],
) -> set[str]:
    decisive_ids = {claim.id for claim in claims if claim.decisive}
    citation_ids = {
        citation
        for validation in validations
        if validation.status == "supported"
        and (not decisive_ids or validation.claim_id in decisive_ids)
        for citation in (validation.winning_evidence_ids or validation.evidence_ids)
    }
    return {item.clause_id for item in evidence if item.citation_id in citation_ids}


def _has_conflict_cue(question: str, task_type: str) -> bool:
    lower = (question or "").lower()
    return task_type == "conflict_and_version_reasoning" and any(
        cue in lower for cue in _CONFLICT_CUES
    )


def build_verification_envelope(
    candidate: CandidateResponse,
    evidence: list[Evidence],
    graph: PropertyGraph,
    *,
    question: str = "",
    task_type: str = "",
    enable_horizon: bool = True,
    enable_vita: bool = False,
    enable_boltzmann_attention: bool = False,
    enable_provenance_robust_selection: bool = False,
    repair_verified_citations: bool = False,
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
            use_provenance_robust_selection=enable_provenance_robust_selection,
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
    retained_claims: list[str] = []
    removed_claim_ids: list[str] = []
    for claim in claims:
        validation = validation_by_id[claim.id]
        if validation.status == "supported":
            retained_claims.append(claim.text)
        else:
            removed_claim_ids.append(claim.id)

    review_reasons: list[str] = []
    advisory_reasons: list[str] = []
    decisive_ids = {claim.id for claim in claims if claim.decisive}
    decisive_failures = [
        claim
        for claim in claims
        if claim.decisive and validation_by_id[claim.id].status != "supported"
    ]
    if decisive_failures:
        _append_unique(review_reasons, "one or more decision-bearing claims were not verified")

    contradicted_decisive = any(
        item.status == "contradicted" and item.claim_id in decisive_ids for item in validations
    )
    contradicted_nondecisive = any(
        item.status == "contradicted" and item.claim_id not in decisive_ids for item in validations
    )
    if contradicted_decisive:
        _append_unique(review_reasons, "counterevidence contradicts a decision-bearing claim")
    elif contradicted_nondecisive:
        _append_unique(advisory_reasons, "counterevidence contradicts a non-decisive explanatory claim")

    relevant_clause_ids = _relevant_clause_ids(claims, validations, effective_evidence)
    unresolved_relevant = any(
        record.unresolved and bool(set(record.loser_clause_ids) & relevant_clause_ids)
        for record in resolutions
    )
    unresolved_other = any(record.unresolved for record in resolutions) and not unresolved_relevant
    if unresolved_relevant:
        _append_unique(review_reasons, "policy precedence could not resolve a conflict touching decisive evidence")
    elif unresolved_other:
        _append_unique(advisory_reasons, "an unresolved conflict remains outside the decisive evidence set")

    incomplete_decisive = any(
        not item.provenance_complete and item.claim_id in decisive_ids for item in validations
    )
    incomplete_other = any(
        not item.provenance_complete and item.claim_id not in decisive_ids for item in validations
    )
    if incomplete_decisive:
        _append_unique(review_reasons, "a decision-bearing claim lacks complete provenance")
    elif incomplete_other:
        _append_unique(advisory_reasons, "a non-decisive claim lacks complete provenance")

    removed_decisive = bool(set(removed_claim_ids) & decisive_ids)
    if removed_decisive:
        _append_unique(review_reasons, "an unsupported decision-bearing claim was removed")
    elif removed_claim_ids:
        _append_unique(advisory_reasons, "unsupported non-decisive claims were removed")

    if horizon_certificate:
        for reason in horizon_certificate.review_reasons:
            _append_unique(review_reasons, reason)
    if vita_certificate:
        for reason in vita_certificate.review_reasons:
            _append_unique(review_reasons, reason)
        for reason in vita_certificate.advisory_reasons:
            _append_unique(advisory_reasons, reason)

    if argumentation_certificate and argumentation_certificate.undecided_argument_ids:
        undecided = set(argumentation_certificate.undecided_argument_ids)
        if undecided & relevant_clause_ids:
            _append_unique(review_reasons, "grounded argumentation leaves decisive evidence undecided")
        else:
            _append_unique(advisory_reasons, "grounded argumentation leaves non-decisive evidence undecided")
    if temporal_certificate and temporal_certificate.retroactive_review_required:
        _append_unique(review_reasons, "policy-version replay detects a more restrictive decision drift")
    if boltzmann_certificate and boltzmann_certificate.residual_probability_mass > 0.35:
        _append_unique(
            advisory_reasons,
            "Boltzmann policy attention leaves probability mass outside the selected clause budget",
        )

    if repair_verified_citations:
        by_citation = {item.citation_id: item for item in effective_evidence}
        repaired: list[str] = []
        claim_by_id = {claim.id: claim for claim in claims}
        for validation in validations:
            if validation.status != "supported":
                continue
            candidates = list(validation.winning_evidence_ids or validation.evidence_ids)
            chosen: list[str] = []
            seen_sources: set[str] = set()
            limit = 2 if claim_by_id.get(validation.claim_id) and claim_by_id[validation.claim_id].decisive else 1
            for citation in candidates:
                item = by_citation.get(citation)
                source = item.source if item else citation
                if citation in chosen:
                    continue
                if not chosen or source not in seen_sources:
                    chosen.append(citation)
                    seen_sources.add(source)
                if len(chosen) >= limit:
                    break
            repaired.extend(chosen)
        accepted_citations = sorted(set(repaired))
    else:
        accepted_citations = sorted({
            citation
            for validation in validations
            if validation.status == "supported"
            for citation in validation.evidence_ids
            if citation in candidate.citations
        })
    evidence_citations = {item.citation_id for item in effective_evidence}
    rejected_citations = sorted({
        citation for citation in candidate.citations if citation not in evidence_citations
    })

    supported_ratio = sum(1 for item in validations if item.status == "supported") / max(1, len(validations))
    provenance_ratio = sum(1 for item in validations if item.provenance_complete) / max(1, len(validations))
    resolution_ratio = (
        sum(1 for item in resolutions if not item.unresolved) / max(1, len(resolutions))
        if resolutions
        else 1.0
    )
    horizon_stability = horizon_certificate.horizon_stability if horizon_certificate else 1.0
    cut_robustness = horizon_certificate.evidence_cut_robustness if horizon_certificate else 1.0
    vita_invariance = (
        1.0
        if vita_certificate and vita_certificate.invariant_under_tested_closure
        else 0.0
        if vita_certificate
        else 1.0
    )
    closure_quality = (
        (1.0 if vita_certificate.closure_converged else 0.0)
        * (1.0 - vita_certificate.residual_risk_mass)
        if vita_certificate
        else 1.0
    )
    argumentation_quality = (
        1.0
        if argumentation_certificate and argumentation_certificate.conflict_free
        else 0.0
        if argumentation_certificate
        else 1.0
    )
    temporal_stability = (
        1.0
        if temporal_certificate and temporal_certificate.stable_across_versions
        else 0.0
        if temporal_certificate
        else 1.0
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

    vita_decision = vita_certificate.final_decision if vita_certificate else candidate.decision
    gated_decision = vita_decision if vita_decision in VALID_DECISIONS else candidate.decision

    conflict_routing = _has_conflict_cue(question, task_type)
    if conflict_routing:
        gated_decision = "needs_review"
        _append_unique(
            review_reasons,
            "the request asks the system to resolve a policy conflict or version choice requiring accountable review",
        )
    elif review_reasons or gated_decision == "unknown":
        gated_decision = "needs_review"

    # Evidence-verification questions often ask whether an unsupported or
    # categorically prohibited proposition is true. Preserve a verified
    # not-allowed result instead of converting a soft audit advisory to review.
    all_decisive_supported = all(
        validation_by_id[claim.id].status == "supported"
        for claim in claims
        if claim.decisive
    )
    if (
        task_type == "hallucination_and_evidence_verification"
        and not review_reasons
        and all_decisive_supported
        and candidate.decision in {"not_allowed", "conditional"}
    ):
        gated_decision = candidate.decision

    normative_review = any(
        _requires_human_review(item.text)
        for item in effective_evidence
        if item.clause_id in relevant_clause_ids
    ) or any(_requires_human_review(text) for text in retained_claims)
    if normative_review:
        if gated_decision in {"allowed", "conditional", "unknown"}:
            gated_decision = "needs_review"
        _append_unique(review_reasons, "verified policy evidence requires human review or approval")

    human_review_required = bool(
        candidate.human_review_required
        or gated_decision == "needs_review"
        or normative_review
        or conflict_routing
    )

    if retained_claims:
        verified_answer = " ".join(retained_claims)
    else:
        verified_answer = "The available evidence is insufficient to verify the requested conclusion."
    if gated_decision == "needs_review" and not _requires_human_review(verified_answer):
        verified_answer = f"{verified_answer} Human review is required."

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
        effective_evidence=[item.to_dict() for item in effective_evidence],
        advisory_reasons=advisory_reasons,
    )
