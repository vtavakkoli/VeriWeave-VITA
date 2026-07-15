from __future__ import annotations

"""Counterfactual evidence-horizon search.

The module searches beyond the initially retrieved subgraph for clauses that are
semantically or structurally relevant and then tests whether adding them changes
claim verification or the gated decision.  It also estimates decision fragility
through a small minimum-evidence-cut search.

The implementation is intentionally deterministic and dependency-free.  It is a
research prototype, not a formal completeness guarantee over arbitrary graphs.
"""

from itertools import combinations
from typing import Iterable

from .graph import PropertyGraph, infer_concepts, infer_modality
from .models import (
    AtomicClaim,
    ClaimValidation,
    EvidenceCutRecord,
    EvidenceHorizonCertificate,
    HorizonCandidateRecord,
    ResolutionRecord,
)
from .precedence import resolve_precedence
from .retrieval import Evidence, evidence_from_node
from .utils import clamp, stable_hash, token_similarity, tokens
from .validators import validate_claims

RISK_EDGES = {"CONTRADICTS", "POTENTIAL_CONFLICT", "OVERRIDES", "VALID_IN", "GOVERNED_BY", "APPLIES_TO", "DERIVED_FROM"}
OPPOSING_MODALITIES = {
    ("permission", "prohibition"),
    ("permission", "obligation"),
    ("prohibition", "permission"),
}


def _status_map(validations: Iterable[ClaimValidation]) -> dict[str, str]:
    return {item.claim_id: item.status for item in validations}


def _gated_decision(
    candidate_decision: str,
    claims: list[AtomicClaim],
    validations: list[ClaimValidation],
    resolutions: list[ResolutionRecord],
) -> str:
    by_id = {item.claim_id: item for item in validations}
    decisive_failure = any(claim.decisive and by_id.get(claim.id) and by_id[claim.id].status != "supported" for claim in claims)
    contradicted = any(item.status == "contradicted" for item in validations)
    unresolved_conflict = any(item.unresolved for item in resolutions)
    if candidate_decision == "unknown" or decisive_failure or contradicted or unresolved_conflict:
        return "needs_review"
    return candidate_decision


def _candidate_path(graph: PropertyGraph, candidate_id: str, seed_ids: set[str]) -> tuple[list[str], int]:
    best_path: list[str] = []
    for seed_id in seed_ids:
        path = graph.shortest_path(seed_id, candidate_id, allowed_types=RISK_EDGES, max_depth=6)
        if path and (not best_path or len(path) < len(best_path)):
            best_path = path
    return best_path or [candidate_id], max(0, len(best_path) - 1) if best_path else 99


def _direct_risk(graph: PropertyGraph, candidate_id: str, seed_ids: set[str]) -> float:
    value = 0.0
    for seed_id in seed_ids:
        for edge in graph.edge_between(candidate_id, seed_id):
            if edge.type == "CONTRADICTS":
                value = max(value, 1.0)
            elif edge.type == "POTENTIAL_CONFLICT":
                value = max(value, 0.88)
            elif edge.type == "OVERRIDES":
                value = max(value, 0.95)
            elif edge.type in {"VALID_IN", "GOVERNED_BY"}:
                value = max(value, 0.72)
            elif edge.type in {"APPLIES_TO", "DERIVED_FROM"}:
                value = max(value, 0.55)
    candidate_doc = graph.document_for_clause(candidate_id)
    candidate_node = graph.nodes.get(candidate_id)
    if candidate_doc and candidate_node:
        for seed_id in seed_ids:
            seed_doc = graph.document_for_clause(seed_id)
            seed_node = graph.nodes.get(seed_id)
            if not seed_doc or not seed_node:
                continue
            if graph.edge_between(candidate_doc.id, seed_doc.id, {"OVERRIDES"}):
                same_section = candidate_node.label.strip().lower() == seed_node.label.strip().lower()
                lexical = token_similarity(
                    str(candidate_node.properties.get("text", "")),
                    str(seed_node.properties.get("text", "")),
                )
                value = max(value, 0.95 if same_section or lexical >= 0.12 else 0.35)
    return value


def _modality_opposition(claims: list[AtomicClaim], evidence: Evidence) -> float:
    evidence_modality = evidence.modality
    claim_modalities = {infer_modality(claim.text) for claim in claims}
    return 1.0 if any((modality, evidence_modality) in OPPOSING_MODALITIES for modality in claim_modalities) else 0.0


def _horizon_candidates(
    question: str,
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    max_candidates: int,
) -> list[Evidence]:
    initial_ids = {item.clause_id for item in evidence}
    seed_ids = set(initial_ids)
    query = " ".join([question] + [claim.text for claim in claims])
    query_concepts = set(infer_concepts(query))
    initial_concepts = {concept for item in evidence for concept in item.concepts}
    max_authority = max((item.authority_rank for item in evidence), default=0)
    initial_has_current = any(item.is_current for item in evidence)

    ranked: list[tuple[float, Evidence]] = []
    for node in graph.nodes.values():
        if node.type != "Clause" or node.id in initial_ids:
            continue
        text = str(node.properties.get("text", ""))
        concepts = set(node.properties.get("concepts", []))
        semantic = token_similarity(query, f"{node.label} {text} {' '.join(concepts)}")
        concept_overlap = len((query_concepts | initial_concepts) & concepts) / max(1, len(query_concepts | initial_concepts))
        path, distance = _candidate_path(graph, node.id, seed_ids)
        graph_risk = _direct_risk(graph, node.id, seed_ids)
        if graph_risk == 0.0 and distance < 99:
            graph_risk = clamp(0.75 - 0.12 * max(0, distance - 1))

        authority_rank = int(node.properties.get("authority_rank", 0))
        authority_uplift = clamp((authority_rank - max_authority) / max(1, authority_rank, max_authority)) if authority_rank > max_authority else 0.0
        is_current = bool(node.properties.get("is_current", False))
        temporal_divergence = 1.0 if is_current and not initial_has_current else 0.0
        candidate = evidence_from_node(
            node,
            0.0,
            path,
            role="horizon-candidate",
            retriever="Evidence Horizon",
        )
        modality_opposition = _modality_opposition(claims, candidate)
        vita_novelty = 1.0 if concepts - initial_concepts else 0.0
        score = clamp(
            0.28 * semantic
            + 0.16 * concept_overlap
            + 0.22 * graph_risk
            + 0.14 * authority_uplift
            + 0.10 * temporal_divergence
            + 0.07 * modality_opposition
            + 0.03 * vita_novelty
        )
        # A graph-risk path can surface lexically weak but structurally decisive clauses.
        if score < 0.08 and graph_risk < 0.70:
            continue
        role = "horizon-counterevidence" if graph_risk >= 0.90 or modality_opposition else "horizon-candidate"
        ranked.append((score, Evidence(**{**candidate.to_dict(), "score": score, "role": role})))

    ranked.sort(key=lambda item: (item[0], item[1].authority_rank, item[1].clause_id), reverse=True)
    return [item for _, item in ranked[:max_candidates]]


def _counterfactual_records(
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    candidates: list[Evidence],
) -> tuple[list[HorizonCandidateRecord], list[ResolutionRecord], list[ClaimValidation], str]:
    baseline_resolutions = resolve_precedence(evidence, graph)
    baseline_validations = validate_claims(claims, evidence, graph, baseline_resolutions)
    baseline_statuses = _status_map(baseline_validations)
    baseline_decision = _gated_decision(candidate_decision, claims, baseline_validations, baseline_resolutions)
    records: list[HorizonCandidateRecord] = []

    for item in candidates:
        augmented = evidence + [item]
        resolutions = resolve_precedence(augmented, graph)
        validations = validate_claims(claims, augmented, graph, resolutions)
        statuses = _status_map(validations)
        decision_after = _gated_decision(candidate_decision, claims, validations, resolutions)
        changed_claim_ids = sorted(claim_id for claim_id, status in statuses.items() if baseline_statuses.get(claim_id) != status)
        severity = {"allowed": 0, "conditional": 1, "unknown": 2, "needs_review": 2, "not_allowed": 3}
        risk_elevating_decision = severity.get(decision_after, 2) > severity.get(baseline_decision, 2)
        risky_status_change = any(
            (baseline_statuses.get(claim_id) == "supported" and statuses.get(claim_id) != "supported")
            or (statuses.get(claim_id) == "contradicted" and baseline_statuses.get(claim_id) != "contradicted")
            for claim_id in changed_claim_ids
        )
        decision_changing = risk_elevating_decision
        # The horizon is fail-safe: permissive evidence that merely relaxes a
        # decision is not counted as a blind spot. It records only evidence
        # that invalidates support, introduces contradiction, or makes the
        # gated decision more restrictive.
        if not risky_status_change and not risk_elevating_decision:
            continue

        graph_risk = _direct_risk(graph, item.clause_id, {seed.clause_id for seed in evidence})
        query = " ".join(claim.text for claim in claims)
        semantic = token_similarity(query, item.text)
        authority_uplift = clamp((item.authority_rank - max((seed.authority_rank for seed in evidence), default=0)) / max(1, item.authority_rank))
        temporal_divergence = 1.0 if item.is_current and not any(seed.is_current for seed in evidence) else 0.0
        modality_opposition = _modality_opposition(claims, item)
        records.append(
            HorizonCandidateRecord(
                clause_id=item.clause_id,
                citation_id=item.citation_id,
                horizon_score=round(item.score, 6),
                semantic_relevance=round(semantic, 6),
                graph_risk=round(graph_risk, 6),
                authority_uplift=round(authority_uplift, 6),
                temporal_divergence=round(temporal_divergence, 6),
                modality_opposition=round(modality_opposition, 6),
                graph_distance=max(0, len(item.graph_path) - 1),
                changed_claim_ids=changed_claim_ids,
                decision_before=baseline_decision,
                decision_after=decision_after,
                status_before=baseline_statuses,
                status_after=statuses,
                decision_changing=decision_changing,
                reason=(
                    "The previously unseen clause makes the gated decision more restrictive."
                    if decision_changing
                    else "The previously unseen clause invalidates or contradicts a previously supported claim."
                ),
            )
        )

    records.sort(key=lambda record: (record.decision_changing, len(record.changed_claim_ids), record.horizon_score), reverse=True)
    return records, baseline_resolutions, baseline_validations, baseline_decision


def _minimal_evidence_cut(
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    baseline_validations: list[ClaimValidation],
    baseline_resolutions: list[ResolutionRecord],
    max_cut_size: int = 3,
) -> EvidenceCutRecord | None:
    """Find the smallest *independent source-group* removal that breaks support.

    Clauses from the same source/version are treated as one correlated evidence
    group.  The cut focuses on supported decisive claims when available, and on
    all supported claims otherwise.  This avoids declaring an answer fragile
    merely because a non-decisive explanatory sentence has one citation.
    """
    baseline_statuses = _status_map(baseline_validations)
    baseline_decision = _gated_decision(candidate_decision, claims, baseline_validations, baseline_resolutions)
    decisive_ids = {claim.id for claim in claims if claim.decisive}
    supported_ids = {
        validation.claim_id for validation in baseline_validations if validation.status == "supported"
    }
    target_ids = (decisive_ids & supported_ids) or supported_ids
    if not target_ids:
        return None

    support_ids = {
        citation
        for validation in baseline_validations
        if validation.claim_id in target_ids and validation.status == "supported"
        for citation in (validation.winning_evidence_ids or validation.evidence_ids)
    }
    relevant = [item for item in evidence if item.citation_id in support_ids]
    if not relevant:
        return None

    groups: dict[str, list[Evidence]] = {}
    for item in relevant:
        # Source is the independence unit. Version remains in the key only when
        # source metadata is absent, so multiple versions from one policy do not
        # masquerade as independent corroboration.
        key = item.source or f"{item.citation_id}@{item.version}"
        groups.setdefault(key, []).append(item)

    group_items = list(groups.items())
    for cut_size in range(1, min(max_cut_size, len(group_items)) + 1):
        for subset in combinations(group_items, cut_size):
            removed_clause_ids = {
                item.clause_id
                for _group, items in subset
                for item in items
            }
            remaining = [item for item in evidence if item.clause_id not in removed_clause_ids]
            resolutions = resolve_precedence(remaining, graph)
            validations = validate_claims(claims, remaining, graph, resolutions)
            statuses = _status_map(validations)
            decision_after = _gated_decision(candidate_decision, claims, validations, resolutions)
            changed = sorted(
                claim_id
                for claim_id in target_ids
                if baseline_statuses.get(claim_id) == "supported"
                and statuses.get(claim_id) != "supported"
            )
            if decision_after != baseline_decision or changed:
                citations = sorted(
                    item.citation_id
                    for _group, items in subset
                    for item in items
                )
                return EvidenceCutRecord(
                    removed_citation_ids=citations,
                    changed_claim_ids=changed,
                    decision_before=baseline_decision,
                    decision_after=decision_after,
                    cut_size=cut_size,
                    explanation=(
                        f"Removing {cut_size} independent source group(s) changes the decision "
                        "or invalidates a supported decision-bearing claim."
                    ),
                )
    return None


def analyze_evidence_horizon(
    question: str,
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    *,
    max_candidates: int = 12,
    max_expand: int = 4,
    max_cut_size: int = 3,
) -> tuple[EvidenceHorizonCertificate, list[Evidence], list[ResolutionRecord], list[ClaimValidation]]:
    """Discover unseen evidence, test decision impact, and certify fragility."""
    candidates = _horizon_candidates(question, claims, evidence, graph, max_candidates)
    records, baseline_resolutions, baseline_validations, baseline_decision = _counterfactual_records(
        claims, evidence, graph, candidate_decision, candidates
    )

    candidate_by_id = {item.clause_id: item for item in candidates}
    impactful = [record for record in records if record.decision_changing]
    verification_only = [record for record in records if not record.decision_changing]
    selected_records = (impactful + verification_only)[:max_expand]
    expanded = list(evidence)
    existing = {item.clause_id for item in expanded}
    for record in selected_records:
        item = candidate_by_id[record.clause_id]
        if item.clause_id not in existing:
            expanded.append(item)
            existing.add(item.clause_id)

    expanded_resolutions = resolve_precedence(expanded, graph)
    expanded_validations = validate_claims(claims, expanded, graph, expanded_resolutions)
    expanded_decision = _gated_decision(candidate_decision, claims, expanded_validations, expanded_resolutions)
    cut = _minimal_evidence_cut(
        claims,
        expanded,
        graph,
        candidate_decision,
        expanded_validations,
        expanded_resolutions,
        max_cut_size=max_cut_size,
    )

    decisive_ids = {claim.id for claim in claims if claim.decisive}
    supported_ids = {
        validation.claim_id for validation in expanded_validations if validation.status == "supported"
    }
    target_ids = (decisive_ids & supported_ids) or supported_ids
    support_citations = {
        citation
        for validation in expanded_validations
        if validation.claim_id in target_ids
        for citation in (validation.winning_evidence_ids or validation.evidence_ids)
    }
    support_sources = {
        item.source or f"{item.citation_id}@{item.version}"
        for item in expanded
        if item.citation_id in support_citations
    }
    cut_assessable = bool(target_ids and support_sources)
    if not cut_assessable:
        cut_robustness = 0.0
    else:
        redundancy_floor = clamp((len(support_sources) - 1) / 2.0)
        if cut is None:
            cut_score = 1.0
        elif cut.cut_size <= 1:
            cut_score = 0.0
        else:
            cut_score = clamp((cut.cut_size - 1) / max(1, max_cut_size - 1))
        cut_robustness = min(cut_score, redundancy_floor if len(support_sources) < 3 else 1.0)

    blind_spots = records
    decision_changing = [record for record in records if record.decision_changing]
    horizon_stability = 1.0 / (1.0 + len(decision_changing))
    unseen_evidence_rate = len(blind_spots) / max(1, len(candidates))
    review_reasons: list[str] = []
    if decision_changing:
        review_reasons.append("evidence-horizon search found unseen clauses that change the gated decision")
    if not cut_assessable:
        review_reasons.append("evidence-cut robustness is not assessable because no independently supported claim was found")
    elif cut is not None and cut.cut_size == 1:
        review_reasons.append("the decision depends on a single removable evidence source group")
    if any(record.unresolved for record in expanded_resolutions):
        review_reasons.append("horizon expansion exposes an unresolved precedence conflict")
    if any(item.status == "contradicted" for item in expanded_validations):
        review_reasons.append("horizon expansion exposes counterevidence that contradicts a claim")

    digest_material = "|".join(
        sorted([item.clause_id for item in evidence] + [record.clause_id for record in records])
    )
    graph_digest = stable_hash(digest_material, 16)
    certificate_id = f"horizon:{stable_hash(question + candidate_decision + graph_digest)}"
    certificate = EvidenceHorizonCertificate(
        certificate_id=certificate_id,
        initial_evidence_ids=sorted(item.citation_id for item in evidence),
        scanned_clause_ids=sorted(item.clause_id for item in candidates),
        blind_spots=[record.to_dict() for record in blind_spots],
        decision_changing_blind_spots=[record.to_dict() for record in decision_changing],
        expanded_evidence_ids=sorted(item.citation_id for item in expanded if item.clause_id not in {seed.clause_id for seed in evidence}),
        baseline_decision=baseline_decision,
        expanded_decision=expanded_decision,
        baseline_statuses=_status_map(baseline_validations),
        expanded_statuses=_status_map(expanded_validations),
        minimal_evidence_cut=cut.to_dict() if cut else None,
        evidence_cut_robustness=round(cut_robustness, 6),
        horizon_stability=round(horizon_stability, 6),
        unseen_evidence_rate=round(unseen_evidence_rate, 6),
        requires_review=bool(review_reasons),
        review_reasons=review_reasons,
        graph_digest=graph_digest,
    )
    return certificate, expanded, expanded_resolutions, expanded_validations
