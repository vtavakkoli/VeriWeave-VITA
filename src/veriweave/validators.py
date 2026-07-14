from __future__ import annotations

from .graph import PropertyGraph, infer_modality
from .models import AtomicClaim, ClaimValidation, ResolutionRecord
from .precedence import winning_clause_ids
from .retrieval import Evidence
from .utils import clamp, containment, token_similarity, tokens

NEGATION_MARKERS = {"not", "no", "never", "prohibited", "forbidden", "without", "cannot"}


def _negated(text: str) -> bool:
    return bool(tokens(text, remove_stopwords=False) & NEGATION_MARKERS)


def _support_score(claim: str, evidence: Evidence) -> float:
    score = 0.58 * containment(claim, evidence.text) + 0.42 * token_similarity(claim, evidence.text)
    claim_modality = infer_modality(claim)
    incompatible = (
        claim_modality == "permission" and evidence.modality in {"prohibition", "obligation"}
    ) or (
        claim_modality == "prohibition" and evidence.modality == "permission"
    )
    return score * (0.35 if incompatible else 1.0)


def _contradiction_score(claim: str, evidence: Evidence, graph: PropertyGraph) -> float:
    lexical = token_similarity(claim, evidence.text)
    polarity_mismatch = _negated(claim) != _negated(evidence.text)
    graph_conflict = any(edge.type in {"CONTRADICTS", "POTENTIAL_CONFLICT"} for edge in graph.outgoing.get(evidence.clause_id, []))
    role_conflict = evidence.role == "counterevidence"
    claim_modality = infer_modality(claim)
    modality_conflict = (
        claim_modality == "permission" and evidence.modality in {"prohibition", "obligation"}
    ) or (
        claim_modality == "prohibition" and evidence.modality == "permission"
    )
    return clamp(
        (0.55 * lexical if polarity_mismatch else 0.0)
        + (0.45 * lexical if modality_conflict else 0.0)
        + (0.20 if graph_conflict else 0.0)
        + (0.10 if role_conflict else 0.0)
    )


def validate_claim(
    claim: AtomicClaim,
    evidence: list[Evidence],
    graph: PropertyGraph,
    resolutions: list[ResolutionRecord] | None = None,
) -> ClaimValidation:
    resolutions = resolutions or []
    winners = winning_clause_ids(evidence, resolutions)
    ranked = sorted(((_support_score(claim.text, item), item) for item in evidence), key=lambda item: item[0], reverse=True)
    best_support = ranked[0][0] if ranked else 0.0
    threshold = max(0.19, best_support * 0.70)
    supporting = [item for score, item in ranked if score >= threshold and item.clause_id in winners][:3]

    counter_ranked = sorted(((_contradiction_score(claim.text, item, graph), item) for item in evidence), key=lambda item: item[0], reverse=True)
    contradiction = counter_ranked[0][0] if counter_ranked else 0.0
    counterevidence = [item for score, item in counter_ranked if score >= 0.38][:3]

    claim_tokens = tokens(claim.text)
    concept_tokens: set[str] = set()
    for item in supporting:
        concept_tokens |= set(item.concepts)
        concept_tokens |= tokens(" ".join(item.concepts))
    applicability = len(claim_tokens & concept_tokens) / max(1, min(6, len(claim_tokens)))
    applicability = clamp(applicability + (0.30 if supporting else 0.0))

    version_sensitive = bool(claim_tokens & {"current", "newer", "version", "v1", "v2", "supersedes"})
    current_support = any(item.is_current for item in supporting)
    temporal = 1.0 if not version_sensitive else (1.0 if current_support else 0.0)

    provenance = bool(supporting) and all(item.source and item.version and item.citation_id and item.graph_path for item in supporting)
    unresolved_conflict = any(record.unresolved for record in resolutions if set(record.loser_clause_ids) & {item.clause_id for item in supporting + counterevidence})

    reasons: list[str] = []
    if best_support < 0.19:
        reasons.append("no sufficiently similar supporting clause")
    if contradiction >= 0.45:
        reasons.append("counterevidence or a graph conflict challenges the claim")
    if applicability < 0.30:
        reasons.append("claim applicability is weakly connected to typed concepts")
    if temporal < 1.0:
        reasons.append("version-sensitive claim is not grounded in current evidence")
    if not provenance:
        reasons.append("source, version, citation, or graph-path provenance is incomplete")
    if unresolved_conflict:
        reasons.append("conflicting evidence remains unresolved after precedence analysis")

    if contradiction >= 0.45 and not supporting:
        status = "contradicted"
    elif best_support >= 0.19 and supporting and provenance and temporal == 1.0 and not unresolved_conflict:
        status = "supported"
    elif contradiction >= 0.55:
        status = "contradicted"
    else:
        status = "unresolved"

    return ClaimValidation(
        claim_id=claim.id,
        claim=claim.text,
        status=status,
        support_score=round(best_support, 6),
        contradiction_score=round(contradiction, 6),
        applicability_score=round(applicability, 6),
        temporal_validity_score=round(temporal, 6),
        provenance_complete=provenance,
        evidence_ids=[item.citation_id for item in supporting],
        counterevidence_ids=[item.citation_id for item in counterevidence],
        winning_evidence_ids=[item.citation_id for item in supporting if item.clause_id in winners],
        reasons=reasons,
    )


def validate_claims(
    claims: list[AtomicClaim],
    evidence: list[Evidence],
    graph: PropertyGraph,
    resolutions: list[ResolutionRecord] | None = None,
) -> list[ClaimValidation]:
    return [validate_claim(claim, evidence, graph, resolutions) for claim in claims]
