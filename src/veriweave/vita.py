from __future__ import annotations

"""Coalitional counterevidence closure and decision-space certification.

VeriWeave-Horizon tests unseen clauses individually. This module extends that
idea in three directions:

1. it tests coalitions of omitted clauses, exposing interactions that no single
   clause reveals;
2. it derives a two-sided decision space rather than recording only
   restrictive changes; and
3. it iterates expansion until a bounded fixed point is reached.

The implementation is deterministic and budgeted. ``closure_converged`` means
no additional impact was found within the configured candidate and coalition
budgets; it is not a completeness theorem over an arbitrary corpus.
"""

from itertools import combinations
from .argumentation import build_argumentation_certificate
from .boltzmann_policy_attention import attend_policy_candidates, merge_attention_certificates
from .provenance_robust_selection import select_provenance_robust_candidates, merge_selection_certificates
from .graph import PropertyGraph, infer_concepts
from .horizon import _horizon_candidates, _gated_decision, _status_map
from .models import (
    ArgumentationCertificate,
    BoltzmannPolicyAttentionCertificate,
    AtomicClaim,
    ClaimValidation,
    CoalitionRecord,
    VITACertificate,
    ResolutionRecord,
    TemporalDriftCertificate,
)
from .precedence import resolve_precedence, winning_clause_ids
from .retrieval import Evidence
from .temporal import build_temporal_drift_certificate
from .utils import clamp, stable_hash, token_similarity
from .validators import validate_claims

SEVERITY = {"allowed": 0, "conditional": 1, "unknown": 2, "needs_review": 2, "not_allowed": 3}


def _vita_decision(
    candidate_decision: str,
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    validations: list[ClaimValidation],
    resolutions: list[ResolutionRecord],
) -> str:
    """Derive a decision from winning normative evidence after safety gating.

    Unlike the singleton Horizon gate, the VITA needs a two-sided outcome
    space. Once claims are admissible and conflicts are resolved, the modality
    of the winning, claim-relevant clauses can make the provisional decision
    either more restrictive or more permissive.
    """
    gated = _gated_decision(candidate_decision, claims, validations, resolutions)
    if gated == "needs_review":
        return gated
    winners = winning_clause_ids(evidence, resolutions)
    decisive = [claim for claim in claims if claim.decisive] or claims
    relevant: list[Evidence] = []
    for item in evidence:
        if item.clause_id not in winners:
            continue
        for claim in decisive:
            claim_concepts = set(infer_concepts(claim.text))
            concept_match = bool(claim_concepts & set(item.concepts))
            lexical_match = token_similarity(claim.text, item.text) >= 0.18
            if concept_match or lexical_match:
                relevant.append(item)
                break
    modalities = {item.modality for item in relevant}
    if "prohibition" in modalities:
        return "not_allowed"
    if "obligation" in modalities:
        return "conditional"
    if "permission" in modalities:
        return "allowed"
    return gated


def _change_signature(
    baseline_statuses: dict[str, str],
    baseline_decision: str,
    validations: list[ClaimValidation],
    decision: str,
) -> tuple[list[str], bool]:
    statuses = _status_map(validations)
    changed = sorted(
        claim_id for claim_id, status in statuses.items() if baseline_statuses.get(claim_id) != status
    )
    return changed, decision != baseline_decision


def _direction(before: str, after: str) -> str:
    if SEVERITY.get(after, 2) > SEVERITY.get(before, 2):
        return "more_restrictive"
    if SEVERITY.get(after, 2) < SEVERITY.get(before, 2):
        return "more_permissive"
    return "status_only"


def _record_for_subset(
    subset: tuple[Evidence, ...],
    claims: list[AtomicClaim],
    base_evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    baseline_statuses: dict[str, str],
    baseline_decision: str,
    singleton_changed: dict[str, bool],
) -> tuple[CoalitionRecord | None, str, dict[str, str]]:
    augmented = list(base_evidence) + list(subset)
    resolutions = resolve_precedence(augmented, graph)
    validations = validate_claims(claims, augmented, graph, resolutions)
    statuses = _status_map(validations)
    decision = _vita_decision(candidate_decision, claims, augmented, validations, resolutions)
    changed_claims, decision_changed = _change_signature(
        baseline_statuses, baseline_decision, validations, decision
    )
    if not changed_claims and not decision_changed:
        return None, decision, statuses

    synergy = len(subset) > 1 and not any(singleton_changed.get(item.clause_id, False) for item in subset)
    concepts = sorted({concept for item in subset for concept in item.concepts})
    modalities = sorted({item.modality for item in subset})
    precedence = [record.to_dict() for record in resolutions if set(record.loser_clause_ids) & {item.clause_id for item in subset}]
    score = sum(item.score for item in subset) / max(1, len(subset))
    if synergy:
        score = clamp(score + 0.15)
    record = CoalitionRecord(
        clause_ids=sorted(item.clause_id for item in subset),
        citation_ids=sorted(item.citation_id for item in subset),
        coalition_size=len(subset),
        changed_claim_ids=changed_claims,
        decision_before=baseline_decision,
        decision_after=decision,
        status_before=baseline_statuses,
        status_after=statuses,
        risk_direction=_direction(baseline_decision, decision),
        synergy=synergy,
        coalition_score=round(score, 6),
        witness={
            "concept_union": concepts,
            "modalities": modalities,
            "graph_paths": {item.citation_id: item.graph_path for item in subset},
            "precedence_records": precedence,
            "explanation": (
                "The clauses have a joint effect that was absent when each clause was tested alone."
                if synergy
                else "The omitted evidence changes at least one verified claim or the gated decision."
            ),
        },
    )
    return record, decision, statuses


def _evaluate_pool(
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    pool: list[Evidence],
    max_coalition_size: int,
) -> tuple[list[CoalitionRecord], set[str], int]:
    baseline_resolutions = resolve_precedence(evidence, graph)
    baseline_validations = validate_claims(claims, evidence, graph, baseline_resolutions)
    baseline_statuses = _status_map(baseline_validations)
    baseline_decision = _vita_decision(
        candidate_decision, claims, evidence, baseline_validations, baseline_resolutions
    )

    singleton_changed: dict[str, bool] = {}
    records: list[CoalitionRecord] = []
    reachable = {baseline_decision}
    tested = 0

    for item in pool:
        tested += 1
        record, decision, statuses = _record_for_subset(
            (item,),
            claims,
            evidence,
            graph,
            candidate_decision,
            baseline_statuses,
            baseline_decision,
            singleton_changed,
        )
        changed = record is not None
        singleton_changed[item.clause_id] = changed
        reachable.add(decision)
        if record:
            records.append(record)

    for size in range(2, min(max_coalition_size, len(pool)) + 1):
        for subset in combinations(pool, size):
            tested += 1
            record, decision, _statuses = _record_for_subset(
                subset,
                claims,
                evidence,
                graph,
                candidate_decision,
                baseline_statuses,
                baseline_decision,
                singleton_changed,
            )
            reachable.add(decision)
            if record:
                records.append(record)

    records.sort(
        key=lambda item: (
            item.synergy,
            item.risk_direction == "more_restrictive",
            len(item.changed_claim_ids),
            item.coalition_score,
        ),
        reverse=True,
    )
    return records, reachable, tested


def analyze_vita_decision_space(
    question: str,
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    *,
    max_candidates: int = 8,
    max_coalition_size: int = 2,
    max_rounds: int = 3,
    max_expand_per_round: int = 4,
    use_boltzmann_attention: bool = False,
    use_provenance_robust_selection: bool = False,
    boltzmann_max_candidates: int = 12,
    boltzmann_temperature: float = 0.75,
    boltzmann_min_temperature: float = 0.20,
    boltzmann_max_temperature: float = 1.50,
    boltzmann_attention_mass: float = 0.90,
    boltzmann_anneal_rate: float = 0.82,
    boltzmann_size_penalty: float = 0.08,
) -> tuple[
    VITACertificate,
    list[Evidence],
    list[ResolutionRecord],
    list[ClaimValidation],
    ArgumentationCertificate,
    TemporalDriftCertificate,
    BoltzmannPolicyAttentionCertificate | None,
]:
    expanded = list(evidence)
    existing = {item.clause_id for item in expanded}
    all_records: list[CoalitionRecord] = []
    tested_candidate_ids: set[str] = set()
    all_reachable: set[str] = set()
    total_tested = 0
    residual_masses: list[float] = []
    converged = False
    rounds_run = 0
    attention_rounds: list[BoltzmannPolicyAttentionCertificate] = []
    selection_rounds = []

    for round_index in range(1, max_rounds + 1):
        rounds_run = round_index
        # Rank every available VITA clause, then expose only the configured
        # budget. The untested score mass is reported as residual risk.
        all_candidates = _horizon_candidates(
            question, claims, expanded, graph, max_candidates=max(1, len(graph.nodes))
        )
        candidates = [item for item in all_candidates if item.clause_id not in existing]
        if use_provenance_robust_selection:
            selection_universe = candidates[: max(max_candidates, boltzmann_max_candidates)]
            pool, selection_certificate = select_provenance_robust_candidates(
                question,
                claims,
                expanded,
                selection_universe,
                graph,
                selection_budget=max_candidates,
            )
            selection_rounds.append(selection_certificate)
            residual_masses.append(selection_certificate.residual_risk_mass)
        elif use_boltzmann_attention:
            attention_universe = candidates[: max(max_candidates, boltzmann_max_candidates)]
            pool, attention_certificate = attend_policy_candidates(
                question,
                claims,
                expanded,
                attention_universe,
                graph,
                candidate_decision,
                base_temperature=boltzmann_temperature,
                minimum_temperature=boltzmann_min_temperature,
                maximum_temperature=boltzmann_max_temperature,
                max_coalition_size=max_coalition_size,
                selection_budget=max_candidates,
                attention_mass=boltzmann_attention_mass,
                size_penalty=boltzmann_size_penalty,
                round_index=round_index,
                anneal_rate=boltzmann_anneal_rate,
            )
            attention_rounds.append(attention_certificate)
            residual_masses.append(attention_certificate.residual_probability_mass)
        else:
            pool = candidates[:max_candidates]
            total_score = sum(max(0.0, item.score) for item in candidates)
            tested_score = sum(max(0.0, item.score) for item in pool)
            residual_masses.append(0.0 if total_score == 0 else clamp((total_score - tested_score) / total_score))
        tested_candidate_ids.update(item.clause_id for item in pool)
        if not pool:
            converged = True
            break

        records, reachable, tested = _evaluate_pool(
            claims,
            expanded,
            graph,
            candidate_decision,
            pool,
            max_coalition_size,
        )
        total_tested += tested
        all_reachable.update(reachable)
        all_records.extend(records)

        # Expand with clauses responsible for the strongest restrictive or
        # synergistic changes. Permissive-only evidence is certified in the
        # VITA candidate space but is not used to weaken the fail-safe answer.
        selected_ids: list[str] = []
        for record in records:
            if record.risk_direction != "more_restrictive" and not record.synergy:
                continue
            for clause_id in record.clause_ids:
                if clause_id not in existing and clause_id not in selected_ids:
                    selected_ids.append(clause_id)
                if len(selected_ids) >= max_expand_per_round:
                    break
            if len(selected_ids) >= max_expand_per_round:
                break
        if not selected_ids:
            converged = True
            break
        by_id = {item.clause_id: item for item in pool}
        added = 0
        for clause_id in selected_ids:
            item = by_id.get(clause_id)
            if item and clause_id not in existing:
                expanded.append(item)
                existing.add(clause_id)
                added += 1
        if added == 0:
            converged = True
            break

    resolutions = resolve_precedence(expanded, graph)
    validations = validate_claims(claims, expanded, graph, resolutions)
    baseline_resolutions = resolve_precedence(evidence, graph)
    baseline_validations = validate_claims(claims, evidence, graph, baseline_resolutions)
    baseline_decision = _vita_decision(
        candidate_decision, claims, evidence, baseline_validations, baseline_resolutions
    )
    final_decision = _vita_decision(
        candidate_decision, claims, expanded, validations, resolutions
    )
    all_reachable.update({baseline_decision, final_decision})

    ordered = sorted(all_reachable, key=lambda decision: (SEVERITY.get(decision, 2), decision))
    permissive = ordered[0] if ordered else baseline_decision
    restrictive = ordered[-1] if ordered else baseline_decision
    width = SEVERITY.get(restrictive, 2) - SEVERITY.get(permissive, 2)
    synergistic = [record for record in all_records if record.synergy]
    restrictive_records = [record for record in all_records if record.risk_direction == "more_restrictive"]
    status_changes = [record for record in all_records if record.changed_claim_ids]
    review_reasons: list[str] = []
    if synergistic:
        review_reasons.append("coalitional closure found interacting omitted clauses that singleton tests missed")
    if restrictive_records:
        review_reasons.append("the decision space contains a more restrictive reachable decision")
    if width > 0:
        review_reasons.append("the answer is not invariant across the tested evidence search space")
    residual = max(residual_masses, default=0.0)
    residual_threshold = 0.35 if use_boltzmann_attention else (0.30 if use_provenance_robust_selection else 0.25)
    if residual > residual_threshold:
        review_reasons.append("a material fraction of VITA attention or ranked risk remained outside the coalition budget")
    if not converged:
        review_reasons.append("bounded closure stopped before a fixed point was established")

    digest = stable_hash(
        "|".join(sorted(existing))
        + "|"
        + "|".join(
            f"{','.join(record.clause_ids)}:{record.decision_after}:{record.synergy}"
            for record in all_records
        ),
        16,
    )
    selection_certificate = merge_selection_certificates(question, selection_rounds)
    certificate = VITACertificate(
        certificate_id=f"vita:{stable_hash(question + candidate_decision + digest)}",
        baseline_decision=baseline_decision,
        reachable_decisions=ordered,
        most_permissive_decision=permissive,
        most_restrictive_decision=restrictive,
        decision_space_width=width,
        invariant_under_tested_closure=width == 0 and not status_changes,
        tested_subsets=total_tested,
        candidate_clause_ids=sorted(tested_candidate_ids),
        coalitional_blind_spots=[record.to_dict() for record in all_records],
        synergistic_blind_spots=[record.to_dict() for record in synergistic],
        closure_rounds=rounds_run,
        closure_converged=converged,
        residual_risk_mass=round(residual, 6),
        requires_review=bool(review_reasons),
        review_reasons=review_reasons,
        graph_digest=digest,
        selection_certificate=selection_certificate.to_dict() if selection_certificate else None,
    )
    boltzmann_certificate = merge_attention_certificates(question, attention_rounds)
    argumentation = build_argumentation_certificate(expanded, graph, resolutions)
    temporal = build_temporal_drift_certificate(
        question,
        claims,
        graph,
        candidate_decision,
        focus_clause_ids=[item.clause_id for item in expanded],
    )
    return certificate, expanded, resolutions, validations, argumentation, temporal, boltzmann_certificate
