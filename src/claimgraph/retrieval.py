from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .graph import Node, PropertyGraph
from .utils import token_list, tokens


@dataclass(frozen=True)
class Evidence:
    clause_id: str
    citation_id: str
    text: str
    score: float
    source: str
    version: str
    concepts: list[str]
    graph_path: list[str]

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "citation_id": self.citation_id,
            "text": self.text,
            "score": round(self.score, 6),
            "source": self.source,
            "version": self.version,
            "concepts": self.concepts,
            "graph_path": self.graph_path,
        }


class TextRetriever:
    def __init__(self, graph: PropertyGraph):
        self.clauses = [node for node in graph.nodes.values() if node.type == "Clause"]
        self.docs = [token_list(str(node.properties.get("text", "")), remove_stopwords=True) for node in self.clauses]
        self.avg_len = sum(map(len, self.docs)) / max(1, len(self.docs))
        vocab = set().union(*(set(d) for d in self.docs)) if self.docs else set()
        self.idf = {}
        for term in vocab:
            df = sum(1 for doc in self.docs if term in doc)
            self.idf[term] = math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))

    def _score(self, query: str, index: int) -> float:
        doc = self.docs[index]
        tf = Counter(doc)
        k1, b = 1.5, 0.75
        score = 0.0
        for term in token_list(query, remove_stopwords=True):
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * len(doc) / max(1.0, self.avg_len))
            score += self.idf.get(term, 0.0) * (freq * (k1 + 1)) / max(denom, 1e-9)
        return score

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        scored = sorted(((self._score(query, i), node) for i, node in enumerate(self.clauses)), key=lambda x: x[0], reverse=True)
        return [_node_to_evidence(node, score, [node.id]) for score, node in scored[:top_k]]


class GraphRetriever:
    def __init__(self, graph: PropertyGraph):
        self.graph = graph

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        selected = self.graph.ranked_clauses(query, top_k=max(top_k * 2, top_k))
        q = query.lower()
        out: list[Evidence] = []
        seen: set[str] = set()
        for score, clause in selected:
            if clause.id in seen:
                continue
            path = [clause.id]
            adjusted = score
            for edge in self.graph.outgoing.get(clause.id, []):
                if edge.type in {"APPLIES_TO", "DERIVED_FROM", "VALID_IN"}:
                    path.append(edge.target)
                if edge.type == "CONTRADICTS" and any(x in q for x in ["conflict", "contradict", "stricter"]):
                    adjusted += 0.2
                if edge.type == "SUPPORTS":
                    adjusted += 0.03
            # Follow document-level OVERRIDES edges for version-sensitive questions.
            for edge in list(self.graph.outgoing.get(clause.id, [])):
                if edge.type == "DERIVED_FROM":
                    for doc_edge in self.graph.outgoing.get(edge.target, []):
                        if doc_edge.type == "OVERRIDES":
                            path.extend([edge.target, doc_edge.target])
                            if any(x in q for x in ["current", "newer", "version", "v2"]):
                                adjusted += 0.25
            out.append(_node_to_evidence(clause, adjusted, path))
            seen.add(clause.id)
            if len(out) >= top_k:
                break
        return out


def _node_to_evidence(node: Node, score: float, path: list[str]) -> Evidence:
    props = node.properties
    return Evidence(
        clause_id=node.id,
        citation_id=str(props.get("citation_id", node.id)),
        text=str(props.get("text", "")),
        score=float(score),
        source=str(props.get("source", "")),
        version=str(props.get("version", "")),
        concepts=list(props.get("concepts", [])),
        graph_path=path,
    )
