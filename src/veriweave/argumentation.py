from __future__ import annotations

"""Argumentation certificate over evidence clauses.

Each evidence clause is treated as an argument. Typed graph conflicts create
attacks; precedence resolutions turn attacks into directed defeats. The
certificate computes the grounded extension, which is unique and skeptical.
This is intentionally a compact research implementation rather than a full
ASPIC+ or defeasible-logic engine.
"""

from .graph import PropertyGraph, modalities_opposed
from .models import ArgumentationCertificate, ResolutionRecord
from .retrieval import Evidence
from .utils import stable_hash


def build_argumentation_certificate(
    evidence: list[Evidence],
    graph: PropertyGraph,
    resolutions: list[ResolutionRecord],
) -> ArgumentationCertificate:
    ids = {item.clause_id for item in evidence}
    attacks: set[tuple[str, str, str]] = set()

    items = sorted(evidence, key=lambda item: item.clause_id)
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            graph_conflict = bool(
                graph.edge_between(left.clause_id, right.clause_id, {"CONTRADICTS", "POTENTIAL_CONFLICT"})
            )
            shared_concepts = bool(set(left.concepts) & set(right.concepts))
            modality_conflict = shared_concepts and modalities_opposed(left.modality, right.modality)
            if graph_conflict or modality_conflict:
                basis = "typed-graph-conflict" if graph_conflict else "opposing-normative-modalities"
                attacks.add((left.clause_id, right.clause_id, basis))
                attacks.add((right.clause_id, left.clause_id, basis))

    defeats: set[tuple[str, str, str]] = set()
    resolved_pairs: set[frozenset[str]] = set()
    for record in resolutions:
        participants = set(record.loser_clause_ids)
        if record.winner_clause_id:
            participants.add(record.winner_clause_id)
        participants &= ids
        if len(participants) < 2:
            continue
        resolved_pairs.add(frozenset(participants))
        if record.winner_clause_id and record.winner_clause_id in ids:
            for loser in record.loser_clause_ids:
                if loser in ids:
                    defeats.add((record.winner_clause_id, loser, record.basis))
        else:
            participants_list = sorted(participants)
            for left in participants_list:
                for right in participants_list:
                    if left != right:
                        defeats.add((left, right, record.basis))

    # Attacks not addressed by precedence remain mutual defeats. This is a
    # skeptical choice: neither side becomes accepted without a defender.
    for source, target, basis in attacks:
        if frozenset((source, target)) not in resolved_pairs:
            defeats.add((source, target, basis))

    attackers: dict[str, set[str]] = {argument_id: set() for argument_id in ids}
    targets: dict[str, set[str]] = {argument_id: set() for argument_id in ids}
    for source, target, _basis in defeats:
        if source in ids and target in ids:
            attackers[target].add(source)
            targets[source].add(target)

    accepted: set[str] = set()
    iterations = 0
    while True:
        iterations += 1
        defended: set[str] = set()
        for argument_id in ids:
            # An argument is defended by S when every attacker is attacked by S.
            if all(any(attacker in targets[defender] for defender in accepted) for attacker in attackers[argument_id]):
                defended.add(argument_id)
        if defended == accepted:
            break
        accepted = defended
        if iterations > len(ids) + 1:
            break

    rejected = {target for source in accepted for target in targets[source]}
    undecided = ids - accepted - rejected
    conflict_free = not any(source in accepted and target in accepted for source, target, _ in defeats)

    attack_payload = [
        {"source": source, "target": target, "basis": basis}
        for source, target, basis in sorted(attacks)
    ]
    defeat_payload = [
        {"source": source, "target": target, "basis": basis}
        for source, target, basis in sorted(defeats)
    ]
    digest = stable_hash(
        "|".join(sorted(ids)) + "|" + "|".join(f"{s}>{t}:{b}" for s, t, b in sorted(defeats)),
        16,
    )
    return ArgumentationCertificate(
        certificate_id=f"argumentation:{stable_hash(digest)}",
        accepted_argument_ids=sorted(accepted),
        rejected_argument_ids=sorted(rejected),
        undecided_argument_ids=sorted(undecided),
        attack_edges=attack_payload,
        defeat_edges=defeat_payload,
        grounded_iterations=iterations,
        conflict_free=conflict_free,
        graph_digest=digest,
    )
