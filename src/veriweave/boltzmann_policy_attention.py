from __future__ import annotations

"""Inference-time Boltzmann Policy Attention over omitted policy clauses.

The mechanism is deliberately distinct from token-level Transformer attention.
It defines a bounded energy model over *sets of policy clauses*. Unary terms
measure the decision-conditioned value of inspecting an omitted clause, while
pairwise couplings model interactions such as conflict, override, version
succession, complementary scope, and redundancy. Exact enumeration is used for
small candidate pools, making the certificate deterministic and auditable.
"""

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable

from .graph import PropertyGraph, infer_concepts, modalities_opposed
from .models import AtomicClaim, BoltzmannPolicyAttentionCertificate
from .retrieval import Evidence
from .utils import clamp, stable_hash, token_similarity


@dataclass(frozen=True)
class _CandidateEnergy:
    clause_id: str
    citation_id: str
    utility: float
    energy: float
    semantic: float
    risk_alignment: float
    authority: float
    temporal: float
    graph_structural: float
    novelty: float

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "citation_id": self.citation_id,
            "utility": round(self.utility, 6),
            "energy": round(self.energy, 6),
            "semantic": round(self.semantic, 6),
            "risk_alignment": round(self.risk_alignment, 6),
            "authority": round(self.authority, 6),
            "temporal": round(self.temporal, 6),
            "graph_structural": round(self.graph_structural, 6),
            "novelty": round(self.novelty, 6),
        }


def _decision_risk_alignment(decision: str, modality: str) -> float:
    """Two-sided attention: inspect evidence that can tighten or relax a decision."""
    table = {
        "allowed": {"prohibition": 1.00, "obligation": 0.85, "permission": 0.15, "guidance": 0.25},
        "conditional": {"prohibition": 0.95, "obligation": 0.65, "permission": 0.55, "guidance": 0.25},
        "not_allowed": {"permission": 1.00, "obligation": 0.70, "prohibition": 0.20, "guidance": 0.25},
        "needs_review": {"prohibition": 0.85, "permission": 0.85, "obligation": 0.70, "guidance": 0.35},
        "unknown": {"prohibition": 0.85, "permission": 0.85, "obligation": 0.70, "guidance": 0.35},
    }
    return table.get(decision, table["unknown"]).get(modality, 0.25)


def _structural_signal(graph: PropertyGraph, clause_id: str, seed_ids: set[str]) -> float:
    if not seed_ids:
        return 0.35
    direct_types = {"CONTRADICTS", "POTENTIAL_CONFLICT", "OVERRIDES", "SUPERSEDES", "APPLIES_TO"}
    direct = 0.0
    for seed in seed_ids:
        edges = graph.edge_between(clause_id, seed)
        for edge in edges:
            if edge.type in direct_types:
                direct = max(direct, 1.0 if edge.type in {"OVERRIDES", "CONTRADICTS"} else 0.85)
    if direct:
        return direct
    best = 99
    for seed in seed_ids:
        path = graph.shortest_path(clause_id, seed, max_depth=6)
        if path:
            best = min(best, len(path) - 1)
    if best == 99:
        return 0.0
    return clamp(1.0 - 0.15 * max(0, best - 1))


def _candidate_energy(
    question: str,
    claims: list[AtomicClaim],
    initial_evidence: list[Evidence],
    item: Evidence,
    graph: PropertyGraph,
    candidate_decision: str,
) -> _CandidateEnergy:
    query = " ".join([question] + [claim.text for claim in claims])
    semantic = max(item.score, token_similarity(query, item.text))
    risk_alignment = _decision_risk_alignment(candidate_decision, item.modality)
    max_authority = max((seed.authority_rank for seed in initial_evidence), default=max(1, item.authority_rank))
    authority = clamp(item.authority_rank / max(1, max(max_authority, item.authority_rank)))
    temporal = 1.0 if item.is_current else 0.25
    seed_ids = {seed.clause_id for seed in initial_evidence}
    graph_structural = _structural_signal(graph, item.clause_id, seed_ids)
    initial_concepts = {concept for seed in initial_evidence for concept in seed.concepts}
    item_concepts = set(item.concepts) or set(infer_concepts(item.text))
    novelty = 1.0 if item_concepts - initial_concepts else 0.25
    utility = clamp(
        0.30 * semantic
        + 0.22 * risk_alignment
        + 0.14 * authority
        + 0.12 * temporal
        + 0.17 * graph_structural
        + 0.05 * novelty
    )
    return _CandidateEnergy(
        clause_id=item.clause_id,
        citation_id=item.citation_id,
        utility=utility,
        energy=-utility,
        semantic=semantic,
        risk_alignment=risk_alignment,
        authority=authority,
        temporal=temporal,
        graph_structural=graph_structural,
        novelty=novelty,
    )


def _coupling(left: Evidence, right: Evidence, graph: PropertyGraph) -> tuple[float, list[str]]:
    """Return an Ising-style pairwise utility and its auditable reasons."""
    reasons: list[str] = []
    value = 0.0
    edges = graph.edge_between(left.clause_id, right.clause_id)
    edge_types = {edge.type for edge in edges}
    if "OVERRIDES" in edge_types or "SUPERSEDES" in edge_types:
        value += 0.42
        reasons.append("version-or-override interaction")
    if "CONTRADICTS" in edge_types:
        value += 0.40
        reasons.append("explicit contradiction")
    if "POTENTIAL_CONFLICT" in edge_types:
        value += 0.30
        reasons.append("potential normative conflict")

    left_concepts, right_concepts = set(left.concepts), set(right.concepts)
    union = left_concepts | right_concepts
    concept_overlap = len(left_concepts & right_concepts) / max(1, len(union))
    if concept_overlap:
        value += 0.18 * concept_overlap
        reasons.append("shared policy concepts")
    if modalities_opposed(left.modality, right.modality):
        value += 0.24
        reasons.append("opposed normative modalities")
    if left.source == right.source and left.version != right.version:
        value += 0.28
        reasons.append("same-source version pair")
    if left.is_current != right.is_current:
        value += 0.10
        reasons.append("temporal contrast")

    lexical = token_similarity(left.text, right.text)
    if lexical >= 0.72 and left.modality == right.modality and left.version == right.version:
        value -= 0.26 * lexical
        reasons.append("redundancy repulsion")
    return max(-0.50, min(0.90, value)), reasons


def _adaptive_temperature(
    base_temperature: float,
    energies: list[_CandidateEnergy],
    couplings: list[tuple[str, str, float, list[str]]],
    *,
    minimum: float,
    maximum: float,
) -> float:
    if not energies:
        return clamp(base_temperature, minimum, maximum)
    values = [item.utility for item in energies]
    spread = max(values) - min(values)
    interaction_density = sum(1 for _, _, value, _ in couplings if value > 0.15) / max(1, len(couplings))
    # Flat unary scores and dense interactions justify broader exploration.
    uncertainty = clamp((1.0 - spread) * 0.55 + interaction_density * 0.45)
    return clamp(base_temperature * (0.70 + 0.90 * uncertainty), minimum, maximum)


def _coalitions(items: list[Evidence], max_size: int) -> Iterable[tuple[Evidence, ...]]:
    for size in range(1, min(max_size, len(items)) + 1):
        yield from combinations(items, size)


def attend_policy_candidates(
    question: str,
    claims: list[AtomicClaim],
    initial_evidence: list[Evidence],
    candidates: list[Evidence],
    graph: PropertyGraph,
    candidate_decision: str,
    *,
    base_temperature: float = 0.75,
    minimum_temperature: float = 0.20,
    maximum_temperature: float = 1.50,
    max_coalition_size: int = 3,
    selection_budget: int = 8,
    attention_mass: float = 0.90,
    size_penalty: float = 0.08,
    round_index: int = 1,
    anneal_rate: float = 0.82,
) -> tuple[list[Evidence], BoltzmannPolicyAttentionCertificate]:
    """Compute exact Boltzmann attention over a bounded policy-clause pool."""
    candidates = list(candidates)
    energies = [
        _candidate_energy(question, claims, initial_evidence, item, graph, candidate_decision)
        for item in candidates
    ]
    by_id = {item.clause_id: item for item in candidates}
    unary = {item.clause_id: energy.utility for item, energy in zip(candidates, energies)}

    coupling_rows: list[tuple[str, str, float, list[str]]] = []
    coupling_map: dict[frozenset[str], float] = {}
    for left, right in combinations(candidates, 2):
        value, reasons = _coupling(left, right, graph)
        coupling_map[frozenset({left.clause_id, right.clause_id})] = value
        if abs(value) >= 0.02:
            coupling_rows.append((left.clause_id, right.clause_id, value, reasons))

    temperature = _adaptive_temperature(
        base_temperature,
        energies,
        coupling_rows,
        minimum=minimum_temperature,
        maximum=maximum_temperature,
    )
    temperature = max(minimum_temperature, temperature * (anneal_rate ** max(0, round_index - 1)))

    raw: list[tuple[tuple[Evidence, ...], float, float]] = []
    for coalition in _coalitions(candidates, max_coalition_size):
        ids = [item.clause_id for item in coalition]
        utility = sum(unary[item_id] for item_id in ids) / max(1, len(ids))
        pair_values = [
            coupling_map.get(frozenset({left_id, right_id}), 0.0)
            for left_id, right_id in combinations(ids, 2)
        ]
        if pair_values:
            utility += sum(pair_values) / len(pair_values)
        utility -= size_penalty * max(0, len(coalition) - 1)
        energy = -utility
        raw.append((coalition, utility, energy))

    if not raw:
        digest = stable_hash(question + candidate_decision + "empty", 16)
        certificate = BoltzmannPolicyAttentionCertificate(
            certificate_id=f"bpa:{stable_hash(question + digest)}",
            mechanism="Boltzmann Policy Attention",
            temperature_schedule=[round(temperature, 6)],
            candidate_clause_ids=[],
            unary_energies=[],
            pairwise_couplings=[],
            top_coalitions=[],
            marginal_attention={},
            selected_clause_ids=[],
            selected_attention_mass=1.0,
            covered_coalition_probability=1.0,
            residual_probability_mass=0.0,
            normalized_entropy=0.0,
            effective_sample_size=0.0,
            risk_tail_mass=0.0,
            free_energy=0.0,
            graph_digest=digest,
        )
        return [], certificate

    max_logit = max(utility / max(temperature, 1e-9) for _, utility, _ in raw)
    weighted: list[tuple[tuple[Evidence, ...], float, float, float]] = []
    partition = 0.0
    for coalition, utility, energy in raw:
        weight = math.exp(utility / max(temperature, 1e-9) - max_logit)
        partition += weight
        weighted.append((coalition, utility, energy, weight))
    probs = [(coalition, utility, energy, weight / partition) for coalition, utility, energy, weight in weighted]
    probs.sort(key=lambda row: (row[3], row[1], tuple(item.clause_id for item in row[0])), reverse=True)

    marginals = {item.clause_id: 0.0 for item in candidates}
    for coalition, _utility, _energy, probability in probs:
        for item in coalition:
            marginals[item.clause_id] += probability

    # Select the smallest marginally ranked clause set that captures the target
    # probability mass of complete coalitions, subject to a hard evidence budget.
    ranked_ids = sorted(marginals, key=lambda item_id: (marginals[item_id], unary[item_id], item_id), reverse=True)
    selected_ids: list[str] = []
    captured = 0.0
    total_marginal = sum(marginals.values())
    for item_id in ranked_ids:
        if len(selected_ids) >= max(1, selection_budget):
            break
        selected_ids.append(item_id)
        captured = sum(marginals[value] for value in selected_ids) / max(1e-12, total_marginal)
        if captured >= attention_mass:
            break

    selected_set = set(selected_ids)
    covered_coalition_probability = sum(
        probability
        for coalition, _utility, _energy, probability in probs
        if {item.clause_id for item in coalition}.issubset(selected_set)
    )
    selected = [by_id[item_id] for item_id in selected_ids]
    entropy = -sum(probability * math.log(max(probability, 1e-12)) for *_rest, probability in probs)
    normalized_entropy = entropy / max(1e-12, math.log(max(2, len(probs))))
    effective_sample_size = 1.0 / max(1e-12, sum(probability * probability for *_rest, probability in probs))
    risk_ids = {
        energy.clause_id
        for energy in energies
        if energy.risk_alignment >= 0.80
        or energy.graph_structural >= 0.85
        or (energy.temporal >= 0.95 and energy.authority >= 0.90 and energy.semantic >= 0.65)
    }
    risk_tail_mass = sum(
        probability
        for coalition, _utility, _energy, probability in probs
        if any(item.clause_id in risk_ids for item in coalition)
    )
    free_energy = -temperature * (math.log(partition) + max_logit)
    top_coalitions = [
        {
            "clause_ids": [item.clause_id for item in coalition],
            "citation_ids": [item.citation_id for item in coalition],
            "utility": round(utility, 6),
            "energy": round(energy, 6),
            "probability": round(probability, 6),
        }
        for coalition, utility, energy, probability in probs[: min(20, len(probs))]
    ]
    pairwise = [
        {
            "left_clause_id": left,
            "right_clause_id": right,
            "coupling": round(value, 6),
            "reasons": reasons,
        }
        for left, right, value, reasons in sorted(coupling_rows, key=lambda row: abs(row[2]), reverse=True)
    ]
    digest_material = "|".join(sorted(marginals)) + "|" + "|".join(selected_ids) + f"|{temperature:.8f}"
    digest = stable_hash(digest_material, 16)
    certificate = BoltzmannPolicyAttentionCertificate(
        certificate_id=f"bpa:{stable_hash(question + candidate_decision + digest)}",
        mechanism="Boltzmann Policy Attention",
        temperature_schedule=[round(temperature, 6)],
        candidate_clause_ids=sorted(marginals),
        unary_energies=[item.to_dict() for item in energies],
        pairwise_couplings=pairwise,
        top_coalitions=top_coalitions,
        marginal_attention={key: round(value, 6) for key, value in sorted(marginals.items())},
        selected_clause_ids=selected_ids,
        selected_attention_mass=round(captured, 6),
        covered_coalition_probability=round(covered_coalition_probability, 6),
        residual_probability_mass=round(clamp(1.0 - captured), 6),
        normalized_entropy=round(clamp(normalized_entropy), 6),
        effective_sample_size=round(effective_sample_size, 6),
        risk_tail_mass=round(clamp(risk_tail_mass), 6),
        free_energy=round(free_energy, 6),
        graph_digest=digest,
    )
    return selected, certificate


def merge_attention_certificates(
    question: str,
    certificates: list[BoltzmannPolicyAttentionCertificate],
) -> BoltzmannPolicyAttentionCertificate | None:
    if not certificates:
        return None
    selected: list[str] = []
    for certificate in certificates:
        for clause_id in certificate.selected_clause_ids:
            if clause_id not in selected:
                selected.append(clause_id)
    latest = certificates[-1]
    digest = stable_hash("|".join(certificate.graph_digest for certificate in certificates), 16)
    return BoltzmannPolicyAttentionCertificate(
        certificate_id=f"bpa:{stable_hash(question + digest)}",
        mechanism="Annealed Boltzmann Policy Attention",
        temperature_schedule=[value for certificate in certificates for value in certificate.temperature_schedule],
        candidate_clause_ids=sorted({value for certificate in certificates for value in certificate.candidate_clause_ids}),
        unary_energies=[
            {"round": index + 1, **row}
            for index, certificate in enumerate(certificates)
            for row in certificate.unary_energies
        ],
        pairwise_couplings=[
            {"round": index + 1, **row}
            for index, certificate in enumerate(certificates)
            for row in certificate.pairwise_couplings
        ],
        top_coalitions=[
            {"round": index + 1, **row}
            for index, certificate in enumerate(certificates)
            for row in certificate.top_coalitions[:10]
        ],
        marginal_attention=latest.marginal_attention,
        selected_clause_ids=selected,
        selected_attention_mass=latest.selected_attention_mass,
        covered_coalition_probability=latest.covered_coalition_probability,
        residual_probability_mass=latest.residual_probability_mass,
        normalized_entropy=latest.normalized_entropy,
        effective_sample_size=latest.effective_sample_size,
        risk_tail_mass=max(certificate.risk_tail_mass for certificate in certificates),
        free_energy=latest.free_energy,
        graph_digest=digest,
    )
