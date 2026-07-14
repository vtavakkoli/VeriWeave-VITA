from __future__ import annotations

"""Temporal policy replay and decision-drift certification."""

import re
from collections import defaultdict

from .graph import PropertyGraph, infer_concepts
from .models import AtomicClaim, TemporalDriftCertificate, TemporalSnapshotRecord
from .precedence import resolve_precedence
from .retrieval import evidence_from_node
from .utils import stable_hash, token_similarity
from .validators import validate_claims


SEVERITY = {"allowed": 0, "conditional": 1, "unknown": 2, "needs_review": 2, "not_allowed": 3}


def _version_number(value: str) -> int:
    match = re.search(r"(\d+)", value or "")
    return int(match.group(1)) if match else 1


def _document_family(graph: PropertyGraph, clause_id: str) -> str:
    document = graph.document_for_clause(clause_id)
    if not document:
        return clause_id
    title = re.sub(r"\bv\d+\b", "", document.label.lower()).strip()
    title = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    clause = graph.nodes.get(clause_id)
    section = re.sub(r"[^a-z0-9]+", "-", (clause.label if clause else clause_id).lower()).strip("-")
    return f"{title}:{section}"


def _gated(candidate_decision: str, claims: list[AtomicClaim], validations, resolutions, evidence) -> str:
    validation_by_id = {item.claim_id: item for item in validations}
    if any(claim.decisive and validation_by_id.get(claim.id) and validation_by_id[claim.id].status != "supported" for claim in claims):
        return "needs_review"
    if any(item.status == "contradicted" for item in validations):
        return "needs_review"
    if any(item.unresolved for item in resolutions):
        return "needs_review"
    # Temporal replay must detect a normative polarity flip even when lexical
    # overlap causes a generic support scorer to treat the newer clause as
    # topically related. A prohibition that directly matches a permissive
    # decisive claim invalidates an ``allowed`` decision.
    if candidate_decision == "allowed":
        decisive_texts = [claim.text for claim in claims if claim.decisive] or [claim.text for claim in claims]
        if any(
            item.modality == "prohibition"
            and any(token_similarity(text, item.text) >= 0.18 for text in decisive_texts)
            for item in evidence
        ):
            return "needs_review"
    return "needs_review" if candidate_decision == "unknown" else candidate_decision


def build_temporal_drift_certificate(
    question: str,
    claims: list[AtomicClaim],
    graph: PropertyGraph,
    candidate_decision: str,
    focus_clause_ids: list[str] | None = None,
) -> TemporalDriftCertificate:
    focus_clause_ids = focus_clause_ids or []
    concepts = set(infer_concepts(question + " " + " ".join(claim.text for claim in claims)))
    for clause_id in focus_clause_ids:
        node = graph.nodes.get(clause_id)
        if node:
            concepts.update(node.properties.get("concepts", []))

    relevant = []
    for node in graph.nodes.values():
        if node.type != "Clause":
            continue
        node_concepts = set(node.properties.get("concepts", []))
        if concepts and not (concepts & node_concepts) and node.id not in focus_clause_ids:
            continue
        relevant.append(node)

    if not relevant:
        relevant = [graph.nodes[clause_id] for clause_id in focus_clause_ids if clause_id in graph.nodes]

    by_family: dict[str, list] = defaultdict(list)
    versions: set[int] = set()
    for node in relevant:
        version = _version_number(str(node.properties.get("version", "v1")))
        versions.add(version)
        by_family[_document_family(graph, node.id)].append((version, node))
    for family in by_family:
        by_family[family].sort(key=lambda item: (item[0], item[1].id))

    snapshots: list[TemporalSnapshotRecord] = []
    previous_citations: set[str] = set()
    previous_statuses: dict[str, str] = {}
    previous_decision = ""
    drift_events: list[dict] = []

    for version in sorted(versions or {1}):
        selected = []
        for candidates in by_family.values():
            eligible = [item for item in candidates if item[0] <= version]
            if eligible:
                selected.append(eligible[-1][1])
        evidence = [
            evidence_from_node(
                node,
                score=1.0,
                path=[node.id],
                role="temporal-replay",
                retriever="Temporal Decision Replay",
            )
            for node in selected
        ]
        resolutions = resolve_precedence(evidence, graph)
        validations = validate_claims(claims, evidence, graph, resolutions)
        statuses = {item.claim_id: item.status for item in validations}
        decision = _gated(candidate_decision, claims, validations, resolutions, evidence)
        citations = {item.citation_id for item in evidence}
        changed_claims = sorted(
            claim_id for claim_id, status in statuses.items() if previous_statuses and previous_statuses.get(claim_id) != status
        )
        record = TemporalSnapshotRecord(
            snapshot=f"v{version}",
            active_citation_ids=sorted(citations),
            decision=decision,
            claim_statuses=statuses,
            added_citation_ids=sorted(citations - previous_citations),
            removed_citation_ids=sorted(previous_citations - citations),
            changed_claim_ids=changed_claims,
        )
        snapshots.append(record)
        if previous_decision and (decision != previous_decision or changed_claims):
            drift_events.append(
                {
                    "from_snapshot": snapshots[-2].snapshot,
                    "to_snapshot": record.snapshot,
                    "decision_before": previous_decision,
                    "decision_after": decision,
                    "changed_claim_ids": changed_claims,
                    "added_citation_ids": record.added_citation_ids,
                    "removed_citation_ids": record.removed_citation_ids,
                    "risk_direction": (
                        "more_restrictive"
                        if SEVERITY.get(decision, 2) > SEVERITY.get(previous_decision, 2)
                        else "more_permissive"
                        if SEVERITY.get(decision, 2) < SEVERITY.get(previous_decision, 2)
                        else "status_only"
                    ),
                }
            )
        previous_citations = citations
        previous_statuses = statuses
        previous_decision = decision

    earliest = snapshots[0].decision if snapshots else candidate_decision
    latest = snapshots[-1].decision if snapshots else candidate_decision
    retroactive = any(event["risk_direction"] == "more_restrictive" for event in drift_events)
    digest = stable_hash(
        "|".join(f"{record.snapshot}:{record.decision}:{','.join(record.active_citation_ids)}" for record in snapshots),
        16,
    )
    return TemporalDriftCertificate(
        certificate_id=f"temporal:{stable_hash(question + digest)}",
        snapshots=[record.to_dict() for record in snapshots],
        drift_events=drift_events,
        stable_across_versions=not drift_events,
        earliest_decision=earliest,
        latest_decision=latest,
        retroactive_review_required=retroactive,
        graph_digest=digest,
    )
