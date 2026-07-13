from __future__ import annotations

from dataclasses import asdict, dataclass

from .claims import AtomicClaim
from .graph import PropertyGraph
from .retrieval import Evidence
from .utils import containment, token_similarity, tokens


NEGATION_MARKERS = {"not", "no", "never", "prohibited", "forbidden", "without", "cannot"}


@dataclass(frozen=True)
class ClaimValidation:
    claim_id: str
    claim: str
    status: str
    support_score: float
    contradiction_score: float
    applicability_score: float
    temporal_validity_score: float
    provenance_complete: bool
    evidence_ids: list[str]
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _negated(text: str) -> bool:
    return bool(tokens(text, remove_stopwords=False) & NEGATION_MARKERS)


def _contradiction_score(claim: str, evidence: Evidence, graph: PropertyGraph) -> float:
    lexical = token_similarity(claim, evidence.text)
    polarity_mismatch = _negated(claim) != _negated(evidence.text)
    graph_conflict = any(edge.type == "CONTRADICTS" for edge in graph.outgoing.get(evidence.clause_id, []))
    return min(1.0, (0.65 * lexical if polarity_mismatch else 0.0) + (0.35 if graph_conflict else 0.0))


def validate_claim(claim: AtomicClaim, evidence: list[Evidence], graph: PropertyGraph) -> ClaimValidation:
    ranked = sorted(
        ((0.55 * containment(claim.text, item.text) + 0.45 * token_similarity(claim.text, item.text), item) for item in evidence),
        key=lambda x: x[0],
        reverse=True,
    )
    best_support = ranked[0][0] if ranked else 0.0
    supporting = [item for score, item in ranked if score >= max(0.18, best_support * 0.7)][:3]
    contradictions = [_contradiction_score(claim.text, item, graph) for item in evidence]
    contradiction = max(contradictions, default=0.0)

    claim_tokens = tokens(claim.text)
    concept_tokens = set()
    for item in supporting:
        concept_tokens |= set(item.concepts)
        concept_tokens |= tokens(" ".join(item.concepts))
    applicability = len(claim_tokens & concept_tokens) / max(1, min(5, len(claim_tokens)))
    applicability = min(1.0, applicability + (0.35 if supporting else 0.0))

    version_sensitive = bool(claim_tokens & {"current", "newer", "version", "v1", "v2"})
    current_support = any(
        graph.nodes.get(item.clause_id) and graph.nodes[item.clause_id].properties.get("is_current") for item in supporting
    )
    temporal = 1.0 if not version_sensitive else (1.0 if current_support else 0.0)

    provenance = bool(supporting) and all(item.source and item.version and item.citation_id for item in supporting)
    reasons: list[str] = []
    if best_support < 0.18:
        reasons.append("no sufficiently similar supporting clause")
    if contradiction >= 0.45:
        reasons.append("candidate evidence has conflicting polarity or graph contradiction")
    if applicability < 0.35:
        reasons.append("claim applicability is weakly connected to graph concepts")
    if temporal < 1.0:
        reasons.append("version-sensitive claim is not grounded in the current policy version")
    if not provenance:
        reasons.append("source, version, or citation provenance is incomplete")

    if contradiction >= 0.45:
        status = "contradicted"
    elif best_support >= 0.18 and provenance and temporal == 1.0:
        status = "supported"
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
        reasons=reasons,
    )


def validate_claims(claims: list[AtomicClaim], evidence: list[Evidence], graph: PropertyGraph) -> list[ClaimValidation]:
    return [validate_claim(claim, evidence, graph) for claim in claims]
