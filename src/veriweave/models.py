from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_DECISIONS = {"allowed", "not_allowed", "conditional", "needs_review", "unknown"}


@dataclass(frozen=True)
class CandidateResponse:
    answer: str
    decision: str = "unknown"
    citations: list[str] = field(default_factory=list)
    human_review_required: bool = False
    raw: str = ""
    parse_status: str = "parsed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicClaim:
    id: str
    text: str
    kind: str = "assertion"
    polarity: str = "positive"
    decisive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolutionRecord:
    id: str
    winner_clause_id: str | None
    loser_clause_ids: list[str]
    basis: str
    confidence: float
    unresolved: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    counterevidence_ids: list[str]
    winning_evidence_ids: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HorizonCandidateRecord:
    clause_id: str
    citation_id: str
    horizon_score: float
    semantic_relevance: float
    graph_risk: float
    authority_uplift: float
    temporal_divergence: float
    modality_opposition: float
    graph_distance: int
    changed_claim_ids: list[str]
    decision_before: str
    decision_after: str
    status_before: dict[str, str]
    status_after: dict[str, str]
    decision_changing: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCutRecord:
    removed_citation_ids: list[str]
    changed_claim_ids: list[str]
    decision_before: str
    decision_after: str
    cut_size: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceHorizonCertificate:
    certificate_id: str
    initial_evidence_ids: list[str]
    scanned_clause_ids: list[str]
    blind_spots: list[dict[str, Any]]
    decision_changing_blind_spots: list[dict[str, Any]]
    expanded_evidence_ids: list[str]
    baseline_decision: str
    expanded_decision: str
    baseline_statuses: dict[str, str]
    expanded_statuses: dict[str, str]
    minimal_evidence_cut: dict[str, Any] | None
    evidence_cut_robustness: float
    horizon_stability: float
    unseen_evidence_rate: float
    requires_review: bool
    review_reasons: list[str]
    graph_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(frozen=True)
class BoltzmannPolicyAttentionCertificate:
    certificate_id: str
    mechanism: str
    temperature_schedule: list[float]
    candidate_clause_ids: list[str]
    unary_energies: list[dict[str, Any]]
    pairwise_couplings: list[dict[str, Any]]
    top_coalitions: list[dict[str, Any]]
    marginal_attention: dict[str, float]
    selected_clause_ids: list[str]
    selected_attention_mass: float
    covered_coalition_probability: float
    residual_probability_mass: float
    normalized_entropy: float
    effective_sample_size: float
    risk_tail_mass: float
    free_energy: float
    graph_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationEnvelope:
    envelope_id: str
    candidate_answer: str
    verified_answer: str
    candidate_decision: str
    gated_decision: str
    candidate_review_required: bool
    human_review_required: bool
    confidence: float
    claims: list[dict[str, Any]]
    validations: list[dict[str, Any]]
    resolutions: list[dict[str, Any]]
    accepted_citations: list[str]
    rejected_citations: list[str]
    removed_claim_ids: list[str]
    review_reasons: list[str]
    graph_digest: str
    evidence_horizon: dict[str, Any] | None = None
    vita_certificate: dict[str, Any] | None = None
    argumentation_certificate: dict[str, Any] | None = None
    temporal_drift_certificate: dict[str, Any] | None = None
    boltzmann_policy_attention: dict[str, Any] | None = None
    effective_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CoalitionRecord:
    clause_ids: list[str]
    citation_ids: list[str]
    coalition_size: int
    changed_claim_ids: list[str]
    decision_before: str
    decision_after: str
    status_before: dict[str, str]
    status_after: dict[str, str]
    risk_direction: str
    synergy: bool
    coalition_score: float
    witness: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VITACertificate:
    certificate_id: str
    baseline_decision: str
    reachable_decisions: list[str]
    most_permissive_decision: str
    most_restrictive_decision: str
    decision_space_width: int
    invariant_under_tested_closure: bool
    tested_subsets: int
    candidate_clause_ids: list[str]
    coalitional_blind_spots: list[dict[str, Any]]
    synergistic_blind_spots: list[dict[str, Any]]
    closure_rounds: int
    closure_converged: bool
    residual_risk_mass: float
    requires_review: bool
    review_reasons: list[str]
    graph_digest: str
    selection_certificate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArgumentationCertificate:
    certificate_id: str
    accepted_argument_ids: list[str]
    rejected_argument_ids: list[str]
    undecided_argument_ids: list[str]
    attack_edges: list[dict[str, Any]]
    defeat_edges: list[dict[str, Any]]
    grounded_iterations: int
    conflict_free: bool
    graph_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalSnapshotRecord:
    snapshot: str
    active_citation_ids: list[str]
    decision: str
    claim_statuses: dict[str, str]
    added_citation_ids: list[str]
    removed_citation_ids: list[str]
    changed_claim_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalDriftCertificate:
    certificate_id: str
    snapshots: list[dict[str, Any]]
    drift_events: list[dict[str, Any]]
    stable_across_versions: bool
    earliest_decision: str
    latest_decision: str
    retroactive_review_required: bool
    graph_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
