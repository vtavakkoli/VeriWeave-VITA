from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .data_loader import PolicyDocument
from .utils import stable_hash, token_similarity, tokens


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
    """Small in-memory property graph used by the reproducible prototype.

    The implementation deliberately avoids a graph-database dependency. Its JSON
    export can be imported into Neo4j, Memgraph, or another property-graph store.
    """

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
        if any((e.source, e.target, e.type) == key for e in self.edges):
            return
        self.edges.append(edge)
        self.outgoing.setdefault(edge.source, []).append(edge)
        self.incoming.setdefault(edge.target, []).append(edge)

    def neighbors(self, node_id: str, edge_types: set[str] | None = None) -> list[Node]:
        edges = self.outgoing.get(node_id, []) + self.incoming.get(node_id, [])
        out: list[Node] = []
        for edge in edges:
            if edge_types and edge.type not in edge_types:
                continue
            other = edge.target if edge.source == node_id else edge.source
            if other in self.nodes and self.nodes[other] not in out:
                out.append(self.nodes[other])
        return out

    def find_nodes(self, *, node_type: str | None = None, text: str = "") -> list[Node]:
        query = text.strip().lower()
        found = []
        for node in self.nodes.values():
            if node_type and node.type != node_type:
                continue
            haystack = f"{node.label} {json.dumps(node.properties, ensure_ascii=False)}".lower()
            if not query or query in haystack:
                found.append(node)
        return found

    def ranked_clauses(self, query: str, top_k: int = 6) -> list[tuple[float, Node]]:
        query_tokens = tokens(query)
        scored: list[tuple[float, Node]] = []
        for node in self.nodes.values():
            if node.type != "Clause":
                continue
            text = str(node.properties.get("text", ""))
            concepts = " ".join(str(x) for x in node.properties.get("concepts", []))
            overlap = len(query_tokens & tokens(f"{node.label} {text} {concepts}")) / max(1, len(query_tokens))
            semantic = token_similarity(query, f"{node.label} {text} {concepts}")
            current_bonus = 0.25 if "current" in query.lower() and node.properties.get("is_current") else 0.0
            conflict_bonus = 0.25 if any(x in query.lower() for x in ["conflict", "stricter", "which rule wins"]) and node.properties.get("relation_hint") == "conflict" else 0.0
            score = 0.65 * overlap + 0.35 * semantic + current_bonus + conflict_bonus
            scored.append((score, node))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)[:top_k]

    def explanation_subgraph(self, clause_ids: list[str]) -> dict[str, Any]:
        wanted = set(clause_ids)
        edges = [e for e in self.edges if e.source in wanted or e.target in wanted]
        node_ids = wanted | {e.source for e in edges} | {e.target for e in edges}
        return {
            "nodes": [asdict(self.nodes[n]) for n in sorted(node_ids) if n in self.nodes],
            "edges": [asdict(e) for e in edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {
                "node_types": sorted({n.type for n in self.nodes.values()}),
                "edge_types": sorted({e.type for e in self.edges}),
            },
            "nodes": [asdict(n) for n in sorted(self.nodes.values(), key=lambda x: x.id)],
            "edges": [asdict(e) for e in self.edges],
        }

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


CONCEPT_SYNONYMS: dict[str, set[str]] = {
    "personal-data": {"personal data", "citizen images", "sensitive data", "health data", "benefits data"},
    "external-service": {"external service", "external cloud", "vendor", "cloud service"},
    "human-review": {"human review", "human oversight", "trained human reviewer", "manual review"},
    "auditability": {"audit trail", "logging", "record", "timestamp", "policy hash", "reproduce"},
    "transparency": {"notice", "transparency", "explain", "purpose", "limitations"},
    "access-control": {"role-based access", "least privilege", "authorization", "restricted"},
    "high-impact": {"high-risk", "high impact", "adverse", "eligibility", "public services", "employee evaluation"},
    "policy-version": {"version", "current approved version", "newer policy", "v1", "v2"},
    "rollback": {"rollback", "fallback", "exit planning"},
    "fairness": {"bias", "fairness", "accessibility", "language fairness"},
}


def _concepts(text: str) -> list[str]:
    low = text.lower()
    found = []
    for concept, phrases in CONCEPT_SYNONYMS.items():
        if concept in low or any(p in low for p in phrases):
            found.append(concept)
    return sorted(found)


def _modality(text: str) -> str:
    low = text.lower()
    if "must not" in low or "not allowed" in low or "prohibited" in low:
        return "prohibition"
    if "must" in low or "requires" in low or "required" in low:
        return "obligation"
    if "may" in low or "allowed" in low:
        return "permission"
    return "guidance"


def _relation_hint(section: str, text: str) -> str:
    low = f"{section} {text}".lower()
    if "conflict" in low or "stricter" in low:
        return "conflict"
    if "current" in low or "newer" in low or "supersedes" in low:
        return "version"
    return "support"


def build_property_graph(documents: list[PolicyDocument]) -> PropertyGraph:
    graph = PropertyGraph()
    authority_id = "authority:enterprise-policy-owner"
    graph.add_node(Node(authority_id, "Authority", "Enterprise Policy Owner", {"role": "governance"}))

    clause_by_section: dict[str, list[str]] = {}
    versions_by_title: dict[str, list[tuple[int, str, str]]] = {}

    for document in documents:
        doc_id = f"document:{document.source}"
        graph.add_node(Node(doc_id, "Document", document.title, {
            "source": document.source,
            "version": document.version,
            "domain": document.domain,
            "sha256": document.sha256,
        }))
        version_id = f"version:{document.source}:{document.version}"
        graph.add_node(Node(version_id, "Version", document.version, {"source": document.source}))
        graph.add_edge(Edge(doc_id, version_id, "HAS_VERSION"))
        graph.add_edge(Edge(doc_id, authority_id, "GOVERNED_BY"))

        version_number_match = re.search(r"(\d+)", document.version)
        version_number = int(version_number_match.group(1)) if version_number_match else 1
        normalized_title = re.sub(r"\bv\d+\b", "", document.title.lower()).strip()
        versions_by_title.setdefault(normalized_title, []).append((version_number, document.version, doc_id))

        sections = re.split(r"(?=^##\s+)", document.text, flags=re.M)
        for index, section_text in enumerate(x.strip() for x in sections if x.strip()):
            header = re.search(r"(?m)^##\s+([^\n]+)", section_text)
            if not header:
                continue
            section = header.group(1).strip()
            body = section_text[header.end():].strip()
            clause_id = f"clause:{document.source}:{re.sub(r'[^a-z0-9]+', '-', section.lower()).strip('-')}"
            concepts = _concepts(body)
            node = Node(clause_id, "Clause", section, {
                "text": body,
                "source": document.source,
                "version": document.version,
                "domain": document.domain,
                "citation_id": f"{document.source}#{section}@{document.version}",
                "modality": _modality(body),
                "concepts": concepts,
                "relation_hint": _relation_hint(section, body),
                "is_current": False,
                "sha256": document.sha256[:12],
                "ordinal": index,
            })
            graph.add_node(node)
            graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
            graph.add_edge(Edge(clause_id, version_id, "VALID_IN"))
            clause_by_section.setdefault(section.lower(), []).append(clause_id)
            for concept in concepts:
                concept_id = f"concept:{concept}"
                if concept_id not in graph.nodes:
                    graph.add_node(Node(concept_id, "Concept", concept, {}))
                graph.add_edge(Edge(clause_id, concept_id, "APPLIES_TO"))

    # Mark latest versions and connect version precedence.
    for title, versions in versions_by_title.items():
        versions.sort()
        latest_number = versions[-1][0]
        for number, version, doc_id in versions:
            for edge in graph.incoming.get(doc_id, []):
                if edge.type == "DERIVED_FROM":
                    clause = graph.nodes[edge.source]
                    props = dict(clause.properties)
                    props["is_current"] = number == latest_number
                    graph.nodes[clause.id] = Node(clause.id, clause.type, clause.label, props)
        for previous, current in zip(versions, versions[1:]):
            graph.add_edge(Edge(current[2], previous[2], "OVERRIDES", {"reason": "newer_version"}))

    # Clauses sharing a normalized section can support one another; conflict clauses
    # are explicitly linked to less-strict local guidance where available.
    for section, clause_ids in clause_by_section.items():
        for i, source in enumerate(clause_ids):
            for target in clause_ids[i + 1:]:
                relation = "CONTRADICTS" if section == "policy-conflict" else "SUPPORTS"
                graph.add_edge(Edge(source, target, relation, {"basis": "shared_section"}))
                graph.add_edge(Edge(target, source, relation, {"basis": "shared_section"}))

    return graph
