from __future__ import annotations

import re
from collections import defaultdict

from .graph import PropertyGraph
from .models import ResolutionRecord
from .retrieval import Evidence
from .utils import stable_hash, strip_citations, token_similarity

MODALITY_PRIORITY = {"prohibition": 4, "obligation": 3, "permission": 2, "guidance": 1}
OPPOSING_MODALITIES = {
    ("permission", "prohibition"),
    ("permission", "obligation"),
    ("prohibition", "permission"),
}


def _source_family(source: str) -> str:
    value = (source or "").lower().rsplit("/", 1)[-1]
    value = re.sub(r"\.(md|txt|json|yaml|yml)$", "", value)
    value = re.sub(r"(?:[-_.]v(?:ersion)?\d+(?:[._-]\d+)*)$", "", value)
    value = re.sub(r"(?:[-_.](old|new|legacy|current))$", "", value)
    return value


def _same_section(graph: PropertyGraph, left: Evidence, right: Evidence) -> bool:
    left_node = graph.nodes.get(left.clause_id)
    right_node = graph.nodes.get(right.clause_id)
    if not left_node or not right_node:
        return False
    return left_node.label.strip().lower() == right_node.label.strip().lower()


def _explicit_override(graph: PropertyGraph, left: Evidence, right: Evidence) -> tuple[str | None, str]:
    left_doc = graph.document_for_clause(left.clause_id)
    right_doc = graph.document_for_clause(right.clause_id)
    if not left_doc or not right_doc:
        return None, ""
    for edge in graph.edge_between(left_doc.id, right_doc.id, {"OVERRIDES", "SUPERSEDES"}):
        if edge.source == left_doc.id:
            return left.clause_id, f"explicit {edge.type} edge"
        return right.clause_id, f"explicit {edge.type} edge"
    return None, ""


def _choose_winner(graph: PropertyGraph, left: Evidence, right: Evidence) -> tuple[str | None, str, float]:
    winner, basis = _explicit_override(graph, left, right)
    if winner:
        return winner, basis, 1.0
    if left.is_current != right.is_current:
        return (left.clause_id if left.is_current else right.clause_id), "current-version precedence", 0.95
    if left.authority_rank != right.authority_rank:
        return (
            left.clause_id if left.authority_rank > right.authority_rank else right.clause_id
        ), "authority-rank precedence", 0.90
    left_modality = MODALITY_PRIORITY.get(left.modality, 0)
    right_modality = MODALITY_PRIORITY.get(right.modality, 0)
    if left_modality != right_modality:
        return (
            left.clause_id if left_modality > right_modality else right.clause_id
        ), "conservative modality precedence", 0.75
    if abs(left.score - right.score) >= 0.15:
        return (
            left.clause_id if left.score > right.score else right.clause_id
        ), "retrieval-confidence tie-break", 0.60
    return None, "unresolved equal-precedence conflict", 0.0


def _is_version_alternative(graph: PropertyGraph, left: Evidence, right: Evidence) -> bool:
    if not left.version or not right.version or left.version == right.version:
        return False
    same_family = bool(_source_family(left.source)) and _source_family(left.source) == _source_family(right.source)
    if not same_family:
        return False
    lexical = token_similarity(strip_citations(left.text), strip_citations(right.text))
    return _same_section(graph, left, right) or lexical >= 0.32


def _is_role_conflict(left: Evidence, right: Evidence) -> bool:
    if not any("counterevidence" in role or role == "superseded" for role in (left.role, right.role)):
        return False
    lexical = token_similarity(strip_citations(left.text), strip_citations(right.text))
    modalities_oppose = (left.modality, right.modality) in OPPOSING_MODALITIES or (
        right.modality, left.modality
    ) in OPPOSING_MODALITIES
    return modalities_oppose and lexical >= 0.10 or lexical >= 0.38


def resolve_precedence(evidence: list[Evidence], graph: PropertyGraph) -> list[ResolutionRecord]:
    """Resolve explicit conflicts and genuine alternatives from one policy family.

    Similar clauses from unrelated sources are corroboration, not a version
    conflict. This distinction prevents independent support from being
    incorrectly converted into unresolved precedence records.
    """
    records: list[ResolutionRecord] = []
    visited: set[tuple[str, str]] = set()
    by_concept: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        for concept in item.concepts or ["__untyped__"]:
            by_concept[concept].append(item)

    for items in by_concept.values():
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                pair = tuple(sorted((left.clause_id, right.clause_id)))
                if pair in visited:
                    continue
                graph_conflict = bool(
                    graph.edge_between(
                        left.clause_id,
                        right.clause_id,
                        {"CONTRADICTS", "POTENTIAL_CONFLICT"},
                    )
                )
                explicit_override = _explicit_override(graph, left, right)[0] is not None
                version_alternative = _is_version_alternative(graph, left, right)
                role_conflict = _is_role_conflict(left, right)
                if not (graph_conflict or explicit_override or version_alternative or role_conflict):
                    continue
                visited.add(pair)
                winner, basis, confidence = _choose_winner(graph, left, right)
                losers = [clause_id for clause_id in pair if clause_id != winner] if winner else list(pair)
                records.append(
                    ResolutionRecord(
                        id=f"resolution:{stable_hash('|'.join(pair) + basis)}",
                        winner_clause_id=winner,
                        loser_clause_ids=losers,
                        basis=basis,
                        confidence=confidence,
                        unresolved=winner is None,
                        explanation=(
                            f"{winner} prevails over {', '.join(losers)} because of {basis}."
                            if winner
                            else f"No deterministic precedence rule resolves {pair[0]} against {pair[1]}."
                        ),
                    )
                )
    return records


def winning_clause_ids(evidence: list[Evidence], resolutions: list[ResolutionRecord]) -> set[str]:
    losers = {
        loser
        for record in resolutions
        if not record.unresolved
        for loser in record.loser_clause_ids
    }
    winners = {record.winner_clause_id for record in resolutions if record.winner_clause_id}
    return ({item.clause_id for item in evidence} - losers) | {winner for winner in winners if winner}
