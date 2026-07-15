from __future__ import annotations

"""Deterministic provenance-robust policy selection.

The selector is designed for the evidence-completeness stage of VITA.  Unlike
Boltzmann Policy Attention, it does not spread probability over many nearly
equivalent coalitions.  It greedily maximizes a transparent submodular-style
objective that rewards:

* claim and question coverage;
* complete source/version/path provenance;
* current and higher-authority clauses;
* independent support from distinct sources;
* explicit contradiction, override, and version-risk relations; and
* concept diversity.

Near-duplicate clauses from the same source/version are penalized.  The result
is deterministic for a fixed graph and query, which makes ablations and paired
comparisons easier to interpret.
"""

from dataclasses import asdict, dataclass
from typing import Any

from .graph import PropertyGraph, infer_concepts
from .models import AtomicClaim
from .retrieval import Evidence
from .utils import clamp, stable_hash, token_similarity


RISK_RELATIONS = {"CONTRADICTS", "POTENTIAL_CONFLICT", "OVERRIDES", "SUPERSEDES"}


@dataclass(frozen=True)
class ProvenanceRobustSelectionCertificate:
    certificate_id: str
    mechanism: str
    candidate_clause_ids: list[str]
    selected_clause_ids: list[str]
    selected_citation_ids: list[str]
    objective_score: float
    claim_coverage: float
    concept_coverage: float
    provenance_completeness: float
    current_evidence_rate: float
    source_diversity: float
    independent_support_score: float
    risk_relation_coverage: float
    residual_risk_mass: float
    selected_records: list[dict[str, Any]]
    graph_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _provenance_complete(item: Evidence) -> float:
    fields = [
        bool(item.source),
        bool(item.version),
        bool(item.citation_id),
        bool(item.graph_path),
    ]
    return sum(fields) / len(fields)


def _risk_signal(item: Evidence, selected: list[Evidence], initial: list[Evidence], graph: PropertyGraph) -> float:
    others = initial + selected
    if not others:
        return 0.0
    best = 0.0
    for other in others:
        edge_types = {edge.type for edge in graph.edge_between(item.clause_id, other.clause_id)}
        if "CONTRADICTS" in edge_types:
            best = max(best, 1.0)
        if "OVERRIDES" in edge_types or "SUPERSEDES" in edge_types:
            best = max(best, 0.95)
        if "POTENTIAL_CONFLICT" in edge_types:
            best = max(best, 0.85)
        if item.source == other.source and item.version != other.version:
            best = max(best, 0.75)
        if set(item.concepts) & set(other.concepts) and item.modality != other.modality:
            best = max(best, 0.65)
    return best


def _redundancy(item: Evidence, selected: list[Evidence], initial: list[Evidence]) -> float:
    best = 0.0
    for other in initial + selected:
        lexical = token_similarity(item.text, other.text)
        same_origin = item.source == other.source and item.version == other.version
        penalty = lexical * (1.0 if same_origin else 0.55)
        best = max(best, penalty)
    return clamp(best)


def _claim_relevance(item: Evidence, claims: list[AtomicClaim], question: str) -> tuple[float, list[str]]:
    rows: list[tuple[float, str]] = []
    for claim in claims:
        score = max(
            token_similarity(claim.text, item.text),
            token_similarity(" ".join(infer_concepts(claim.text)), " ".join(item.concepts)),
        )
        rows.append((score, claim.id))
    question_score = token_similarity(question, item.text)
    best = max([question_score] + [score for score, _ in rows], default=0.0)
    covered = [claim_id for score, claim_id in rows if score >= 0.18]
    return clamp(max(best, item.score)), covered


def _independent_support_gain(
    item: Evidence,
    claims: list[AtomicClaim],
    selected: list[Evidence],
    initial: list[Evidence],
) -> float:
    """Reward a new source that can independently support a claim."""
    existing = initial + selected
    gain = 0.0
    for claim in claims:
        item_score = token_similarity(claim.text, item.text)
        if item_score < 0.18:
            continue
        supporting_sources = {
            other.source
            for other in existing
            if other.source and token_similarity(claim.text, other.text) >= 0.18
        }
        if item.source and item.source not in supporting_sources:
            gain += 1.0 if supporting_sources else 0.65
        elif item.source:
            gain += 0.15
    return clamp(gain / max(1, len(claims)))


def select_provenance_robust_candidates(
    question: str,
    claims: list[AtomicClaim],
    initial_evidence: list[Evidence],
    candidates: list[Evidence],
    graph: PropertyGraph,
    *,
    selection_budget: int = 8,
) -> tuple[list[Evidence], ProvenanceRobustSelectionCertificate]:
    candidates = list(candidates)
    selected: list[Evidence] = []
    records: list[dict[str, Any]] = []
    query_concepts = set(infer_concepts(question))
    claim_concepts = {concept for claim in claims for concept in infer_concepts(claim.text)}
    target_concepts = query_concepts | claim_concepts
    covered_claims: set[str] = set()
    covered_concepts = {concept for item in initial_evidence for concept in item.concepts}
    selected_sources = {item.source for item in initial_evidence if item.source}

    remaining = {item.clause_id: item for item in candidates}
    total_positive_potential = 0.0
    candidate_potential: dict[str, float] = {}
    max_authority = max(
        [item.authority_rank for item in initial_evidence + candidates] or [1]
    )

    for item in candidates:
        relevance, _ = _claim_relevance(item, claims, question)
        provenance = _provenance_complete(item)
        temporal = 1.0 if item.is_current else 0.20
        authority = clamp(item.authority_rank / max(1, max_authority))
        risk = _risk_signal(item, [], initial_evidence, graph)
        potential = (
            0.34 * relevance
            + 0.18 * provenance
            + 0.14 * temporal
            + 0.10 * authority
            + 0.16 * risk
            + 0.08 * (1.0 if set(item.concepts) & target_concepts else 0.0)
        )
        candidate_potential[item.clause_id] = max(0.0, potential)
        total_positive_potential += max(0.0, potential)

    while remaining and len(selected) < max(1, selection_budget):
        best_item: Evidence | None = None
        best_row: dict[str, Any] | None = None
        best_score = float("-inf")

        for item in remaining.values():
            relevance, claim_ids = _claim_relevance(item, claims, question)
            new_claims = len(set(claim_ids) - covered_claims) / max(1, len(claims))
            new_concepts_set = set(item.concepts) - covered_concepts
            concept_gain = len(new_concepts_set & target_concepts) / max(1, len(target_concepts))
            provenance = _provenance_complete(item)
            temporal = 1.0 if item.is_current else 0.20
            authority = clamp(item.authority_rank / max(1, max_authority))
            source_gain = 1.0 if item.source and item.source not in selected_sources else 0.15
            independent = _independent_support_gain(item, claims, selected, initial_evidence)
            risk = _risk_signal(item, selected, initial_evidence, graph)
            path_quality = 1.0 if item.graph_path else 0.0
            redundancy = _redundancy(item, selected, initial_evidence)

            marginal = (
                0.22 * relevance
                + 0.14 * new_claims
                + 0.08 * concept_gain
                + 0.12 * provenance
                + 0.09 * temporal
                + 0.06 * authority
                + 0.09 * source_gain
                + 0.12 * independent
                + 0.10 * risk
                + 0.04 * path_quality
                - 0.12 * redundancy
            )
            row = {
                "clause_id": item.clause_id,
                "citation_id": item.citation_id,
                "marginal_score": round(marginal, 6),
                "relevance": round(relevance, 6),
                "new_claim_coverage": round(new_claims, 6),
                "new_concept_coverage": round(concept_gain, 6),
                "provenance": round(provenance, 6),
                "temporal": round(temporal, 6),
                "authority": round(authority, 6),
                "source_diversity_gain": round(source_gain, 6),
                "independent_support_gain": round(independent, 6),
                "risk_relation_signal": round(risk, 6),
                "redundancy_penalty": round(redundancy, 6),
            }
            tie = (marginal, item.is_current, item.authority_rank, item.score, item.clause_id)
            best_tie = (
                best_score,
                best_item.is_current if best_item else False,
                best_item.authority_rank if best_item else -1,
                best_item.score if best_item else -1.0,
                best_item.clause_id if best_item else "",
            )
            if tie > best_tie:
                best_score = marginal
                best_item = item
                best_row = row

        if best_item is None:
            break
        selected.append(best_item)
        records.append(best_row or {})
        remaining.pop(best_item.clause_id, None)
        _, claim_ids = _claim_relevance(best_item, claims, question)
        covered_claims.update(claim_ids)
        covered_concepts.update(best_item.concepts)
        if best_item.source:
            selected_sources.add(best_item.source)

    selected_ids = {item.clause_id for item in selected}
    selected_potential = sum(candidate_potential.get(item_id, 0.0) for item_id in selected_ids)
    residual = 0.0 if total_positive_potential <= 1e-12 else clamp(
        (total_positive_potential - selected_potential) / total_positive_potential
    )
    selected_sources_only = {item.source for item in selected if item.source}
    all_sources = {item.source for item in candidates if item.source}
    claim_coverage = len(covered_claims) / max(1, len(claims))
    concept_coverage = len(covered_concepts & target_concepts) / max(1, len(target_concepts))
    provenance = sum(_provenance_complete(item) for item in selected) / max(1, len(selected))
    current_rate = sum(1 for item in selected if item.is_current) / max(1, len(selected))
    source_diversity = len(selected_sources_only) / max(1, min(len(selected), len(all_sources)))
    independent_support = sum(
        _independent_support_gain(item, claims, selected[:idx], initial_evidence)
        for idx, item in enumerate(selected)
    ) / max(1, len(selected))
    risk_coverage = sum(
        1 for item in selected if _risk_signal(item, selected, initial_evidence, graph) >= 0.65
    ) / max(1, len(selected))
    objective = sum(float(row.get("marginal_score", 0.0)) for row in records)

    digest = stable_hash(
        question
        + "|"
        + "|".join(sorted(item.clause_id for item in initial_evidence))
        + "|"
        + "|".join(item.clause_id for item in selected),
        16,
    )
    certificate = ProvenanceRobustSelectionCertificate(
        certificate_id=f"pro:{stable_hash(question + digest)}",
        mechanism="Provenance-Robust Optimization",
        candidate_clause_ids=sorted(item.clause_id for item in candidates),
        selected_clause_ids=[item.clause_id for item in selected],
        selected_citation_ids=[item.citation_id for item in selected],
        objective_score=round(objective, 6),
        claim_coverage=round(claim_coverage, 6),
        concept_coverage=round(concept_coverage, 6),
        provenance_completeness=round(provenance, 6),
        current_evidence_rate=round(current_rate, 6),
        source_diversity=round(source_diversity, 6),
        independent_support_score=round(independent_support, 6),
        risk_relation_coverage=round(risk_coverage, 6),
        residual_risk_mass=round(residual, 6),
        selected_records=records,
        graph_digest=digest,
    )
    return selected, certificate


def merge_selection_certificates(
    question: str,
    certificates: list[ProvenanceRobustSelectionCertificate],
) -> ProvenanceRobustSelectionCertificate | None:
    if not certificates:
        return None
    selected_clause_ids: list[str] = []
    selected_citation_ids: list[str] = []
    records: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for certificate in certificates:
        candidate_ids.update(certificate.candidate_clause_ids)
        for value in certificate.selected_clause_ids:
            if value not in selected_clause_ids:
                selected_clause_ids.append(value)
        for value in certificate.selected_citation_ids:
            if value not in selected_citation_ids:
                selected_citation_ids.append(value)
        records.extend(certificate.selected_records)
    avg = lambda name: sum(float(getattr(c, name)) for c in certificates) / len(certificates)
    digest = stable_hash(
        question + "|" + "|".join(sorted(candidate_ids)) + "|" + "|".join(selected_clause_ids),
        16,
    )
    return ProvenanceRobustSelectionCertificate(
        certificate_id=f"pro:{stable_hash(question + digest)}",
        mechanism="Provenance-Robust Optimization",
        candidate_clause_ids=sorted(candidate_ids),
        selected_clause_ids=selected_clause_ids,
        selected_citation_ids=selected_citation_ids,
        objective_score=round(sum(c.objective_score for c in certificates), 6),
        claim_coverage=round(avg("claim_coverage"), 6),
        concept_coverage=round(avg("concept_coverage"), 6),
        provenance_completeness=round(avg("provenance_completeness"), 6),
        current_evidence_rate=round(avg("current_evidence_rate"), 6),
        source_diversity=round(avg("source_diversity"), 6),
        independent_support_score=round(avg("independent_support_score"), 6),
        risk_relation_coverage=round(avg("risk_relation_coverage"), 6),
        residual_risk_mass=round(max(c.residual_risk_mass for c in certificates), 6),
        selected_records=records,
        graph_digest=digest,
    )
