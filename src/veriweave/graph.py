from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .data_loader import PolicyDocument
from .utils import token_similarity, tokens


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


class PropertyGraph:
    """Dependency-free typed property graph with deterministic traversal."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(f"Edge refers to unknown node: {edge}")
        key = (edge.source, edge.target, edge.type)
        if any((item.source, item.target, item.type) == key for item in self.edges):
            return
        self.edges.append(edge)
        self.outgoing.setdefault(edge.source, []).append(edge)
        self.incoming.setdefault(edge.target, []).append(edge)

    def neighbors(self, node_id: str, edge_types: set[str] | None = None) -> list[Node]:
        adjacent = self.outgoing.get(node_id, []) + self.incoming.get(node_id, [])
        result: list[Node] = []
        seen: set[str] = set()
        for edge in adjacent:
            if edge_types and edge.type not in edge_types:
                continue
            other = edge.target if edge.source == node_id else edge.source
            if other in self.nodes and other not in seen:
                result.append(self.nodes[other])
                seen.add(other)
        return result

    def edge_between(self, source: str, target: str, edge_types: set[str] | None = None) -> list[Edge]:
        return [
            edge for edge in self.edges
            if ((edge.source == source and edge.target == target) or (edge.source == target and edge.target == source))
            and (not edge_types or edge.type in edge_types)
        ]

    def document_for_clause(self, clause_id: str) -> Node | None:
        for edge in self.outgoing.get(clause_id, []):
            if edge.type == "DERIVED_FROM":
                return self.nodes.get(edge.target)
        return None

    def authority_for_document(self, document_id: str) -> Node | None:
        for edge in self.outgoing.get(document_id, []):
            if edge.type == "GOVERNED_BY":
                return self.nodes.get(edge.target)
        return None

    def shortest_path(self, start: str, goal: str, allowed_types: set[str] | None = None, max_depth: int = 8) -> list[str]:
        if start == goal:
            return [start]
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
        visited = {start}
        while queue:
            node_id, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for edge in self.outgoing.get(node_id, []) + self.incoming.get(node_id, []):
                if allowed_types and edge.type not in allowed_types:
                    continue
                other = edge.target if edge.source == node_id else edge.source
                if other == goal:
                    return path + [other]
                if other not in visited:
                    visited.add(other)
                    queue.append((other, path + [other]))
        return []

    def ranked_clauses(self, query: str, top_k: int = 6) -> list[tuple[float, Node]]:
        query_tokens = tokens(query)
        lower_query = query.lower()
        scored: list[tuple[float, Node]] = []
        for node in self.nodes.values():
            if node.type != "Clause":
                continue
            text = str(node.properties.get("text", ""))
            concepts = " ".join(str(value) for value in node.properties.get("concepts", []))
            overlap = len(query_tokens & tokens(f"{node.label} {text} {concepts}")) / max(1, len(query_tokens))
            semantic = token_similarity(query, f"{node.label} {text} {concepts}")
            current_bonus = 0.20 if any(term in lower_query for term in ("current", "newer", "version")) and node.properties.get("is_current") else 0.0
            conflict_bonus = 0.20 if any(term in lower_query for term in ("conflict", "contradict", "stricter", "which rule wins")) and node.properties.get("relation_hint") == "conflict" else 0.0
            modality_bonus = 0.06 if node.properties.get("modality") in {"prohibition", "obligation"} else 0.0
            score = 0.60 * overlap + 0.34 * semantic + current_bonus + conflict_bonus + modality_bonus
            scored.append((score, node))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)[:top_k]

    def explanation_subgraph(self, clause_ids: list[str], include_claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        wanted = set(clause_ids)
        edges = [edge for edge in self.edges if edge.source in wanted or edge.target in wanted]
        node_ids = wanted | {edge.source for edge in edges} | {edge.target for edge in edges}
        payload = {
            "nodes": [asdict(self.nodes[node_id]) for node_id in sorted(node_ids) if node_id in self.nodes],
            "edges": [asdict(edge) for edge in edges],
        }
        if include_claims:
            payload["claims"] = include_claims
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {
                "node_types": sorted({node.type for node in self.nodes.values()}),
                "edge_types": sorted({edge.type for edge in self.edges}),
            },
            "nodes": [asdict(node) for node in sorted(self.nodes.values(), key=lambda item: item.id)],
            "edges": [asdict(edge) for edge in self.edges],
        }

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


CONCEPT_SYNONYMS: dict[str, set[str]] = {
    "personal-data": {"personal data", "citizen images", "sensitive data", "health data", "benefits data"},
    "external-service": {"external service", "external cloud", "external ai cloud", "vendor", "cloud service"},
    "human-review": {"human review", "human oversight", "trained human reviewer", "manual review"},
    "auditability": {"audit trail", "logging", "record", "timestamp", "policy hash", "reproduce"},
    "transparency": {"notice", "transparency", "explain", "purpose", "limitations"},
    "access-control": {"role-based access", "least privilege", "authorization", "restricted"},
    "high-impact": {"high-risk", "high impact", "adverse", "eligibility", "public services", "employee evaluation"},
    "policy-version": {"version", "current approved version", "newer policy", "v1", "v2"},
    "rollback": {"rollback", "fallback", "exit planning"},
    "fairness": {"bias", "fairness", "accessibility", "language fairness"},
}


def infer_concepts(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for concept, phrases in CONCEPT_SYNONYMS.items():
        if concept in lower or any(phrase in lower for phrase in phrases):
            found.append(concept)
    return sorted(found)


def infer_modality(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("must not", "may not", "not allowed", "prohibit", "prohibits", "prohibited", "forbidden", "cannot")):
        return "prohibition"
    if any(term in lower for term in ("must", "requires", "required")):
        return "obligation"
    if any(term in lower for term in ("may", "allowed", "permitted", "can proceed", "is sufficient", "can ")):
        return "permission"
    return "guidance"


def modalities_opposed(left: str, right: str) -> bool:
    return (left, right) in {
        ("permission", "prohibition"),
        ("permission", "obligation"),
        ("prohibition", "permission"),
    }


def _relation_hint(section: str, text: str) -> str:
    lower = f"{section} {text}".lower()
    if "conflict" in lower or "stricter" in lower:
        return "conflict"
    if any(term in lower for term in ("current", "newer", "supersedes", "version")):
        return "version"
    return "support"


def build_property_graph(documents: list[PolicyDocument]) -> PropertyGraph:
    graph = PropertyGraph()
    authority_nodes: dict[int, str] = {}
    clause_by_section: dict[str, list[str]] = {}
    versions_by_title: dict[str, list[tuple[int, str, str]]] = {}

    for document in documents:
        authority_id = authority_nodes.get(document.authority_rank)
        if authority_id is None:
            authority_id = f"authority:rank-{document.authority_rank}"
            authority_nodes[document.authority_rank] = authority_id
            graph.add_node(Node(authority_id, "Authority", f"Policy authority rank {document.authority_rank}", {"rank": document.authority_rank}))

        doc_id = f"document:{document.source}"
        graph.add_node(Node(doc_id, "Document", document.title, {
            "source": document.source,
            "version": document.version,
            "domain": document.domain,
            "sha256": document.sha256,
            "authority_rank": document.authority_rank,
        }))
        version_id = f"version:{document.source}:{document.version}"
        graph.add_node(Node(version_id, "Version", document.version, {"source": document.source}))
        graph.add_edge(Edge(doc_id, version_id, "HAS_VERSION"))
        graph.add_edge(Edge(doc_id, authority_id, "GOVERNED_BY"))

        match = re.search(r"(\d+)", document.version)
        version_number = int(match.group(1)) if match else 1
        normalized_title = re.sub(r"\bv\d+\b", "", document.title.lower()).strip()
        versions_by_title.setdefault(normalized_title, []).append((version_number, document.version, doc_id))

        sections = re.split(r"(?=^##\s+)", document.text, flags=re.M)
        for ordinal, section_text in enumerate(value.strip() for value in sections if value.strip()):
            header = re.search(r"(?m)^##\s+([^\n]+)", section_text)
            if not header:
                continue
            section = header.group(1).strip()
            body = section_text[header.end():].strip()
            slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
            clause_id = f"clause:{document.source}:{slug}"
            concepts = infer_concepts(body)
            graph.add_node(Node(clause_id, "Clause", section, {
                "text": body,
                "source": document.source,
                "version": document.version,
                "domain": document.domain,
                "citation_id": f"{document.source}#{section}@{document.version}",
                "modality": infer_modality(body),
                "concepts": concepts,
                "relation_hint": _relation_hint(section, body),
                "is_current": False,
                "sha256": document.sha256[:12],
                "ordinal": ordinal,
                "authority_rank": document.authority_rank,
            }))
            graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
            graph.add_edge(Edge(clause_id, version_id, "VALID_IN"))
            clause_by_section.setdefault(section.lower(), []).append(clause_id)
            for concept in concepts:
                concept_id = f"concept:{concept}"
                if concept_id not in graph.nodes:
                    graph.add_node(Node(concept_id, "Concept", concept, {}))
                graph.add_edge(Edge(clause_id, concept_id, "APPLIES_TO"))

    for versions in versions_by_title.values():
        versions.sort()
        latest_number = versions[-1][0]
        for number, _version, doc_id in versions:
            for edge in graph.incoming.get(doc_id, []):
                if edge.type == "DERIVED_FROM":
                    clause = graph.nodes[edge.source]
                    properties = dict(clause.properties)
                    properties["is_current"] = number == latest_number
                    graph.nodes[clause.id] = Node(clause.id, clause.type, clause.label, properties)
        for previous, current in zip(versions, versions[1:]):
            graph.add_edge(Edge(current[2], previous[2], "OVERRIDES", {"reason": "newer_version"}))

    for section, clause_ids in clause_by_section.items():
        for index, source in enumerate(clause_ids):
            for target in clause_ids[index + 1:]:
                relation = "CONTRADICTS" if section == "policy-conflict" else "SUPPORTS"
                graph.add_edge(Edge(source, target, relation, {"basis": "shared_section"}))
                graph.add_edge(Edge(target, source, relation, {"basis": "shared_section"}))

    # Surface normative opposition that ordinary relevance retrieval can miss.
    # The relation is deliberately typed as POTENTIAL_CONFLICT rather than
    # CONTRADICTS because it is generated from shared concepts and opposing
    # modalities and therefore requires downstream precedence validation.
    clauses = [node for node in graph.nodes.values() if node.type == "Clause"]
    for index, left in enumerate(clauses):
        left_concepts = set(left.properties.get("concepts", []))
        left_modality = str(left.properties.get("modality", "guidance"))
        for right in clauses[index + 1:]:
            right_concepts = set(right.properties.get("concepts", []))
            shared = sorted(left_concepts & right_concepts)
            if not shared:
                continue
            right_modality = str(right.properties.get("modality", "guidance"))
            if not modalities_opposed(left_modality, right_modality):
                continue
            lexical = token_similarity(
                str(left.properties.get("text", "")),
                str(right.properties.get("text", "")),
            )
            if lexical < 0.03:
                continue
            properties = {"basis": "shared_concept_opposing_modality", "concepts": shared, "lexical": round(lexical, 6)}
            graph.add_edge(Edge(left.id, right.id, "POTENTIAL_CONFLICT", properties))
            graph.add_edge(Edge(right.id, left.id, "POTENTIAL_CONFLICT", properties))

    return graph
