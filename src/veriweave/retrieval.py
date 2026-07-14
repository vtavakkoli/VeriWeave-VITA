from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass

from .graph import Node, PropertyGraph, infer_concepts
from .utils import clamp, token_list, token_similarity, tokens


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
    role: str = "candidate"
    retriever: str = "unknown"
    modality: str = "guidance"
    is_current: bool = False
    authority_rank: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["score"] = round(self.score, 6)
        return payload


def evidence_from_node(node: Node, score: float, path: list[str], *, role: str, retriever: str) -> Evidence:
    properties = node.properties
    return Evidence(
        clause_id=node.id,
        citation_id=str(properties.get("citation_id", node.id)),
        text=str(properties.get("text", "")),
        score=float(score),
        source=str(properties.get("source", "")),
        version=str(properties.get("version", "")),
        concepts=list(properties.get("concepts", [])),
        graph_path=path,
        role=role,
        retriever=retriever,
        modality=str(properties.get("modality", "guidance")),
        is_current=bool(properties.get("is_current", False)),
        authority_rank=int(properties.get("authority_rank", 0)),
    )


class TextRetriever:
    name = "Text RAG"

    def __init__(self, graph: PropertyGraph):
        self.graph = graph
        self.clauses = [node for node in graph.nodes.values() if node.type == "Clause"]
        self.docs = [token_list(str(node.properties.get("text", "")), remove_stopwords=True) for node in self.clauses]
        self.avg_len = sum(map(len, self.docs)) / max(1, len(self.docs))
        vocabulary = set().union(*(set(doc) for doc in self.docs)) if self.docs else set()
        self.idf: dict[str, float] = {}
        for term in vocabulary:
            document_frequency = sum(1 for document in self.docs if term in document)
            self.idf[term] = math.log(1 + (len(self.docs) - document_frequency + 0.5) / (document_frequency + 0.5))

    def score_node(self, query: str, node: Node) -> float:
        try:
            index = self.clauses.index(node)
        except ValueError:
            return 0.0
        document = self.docs[index]
        frequencies = Counter(document)
        k1, b = 1.5, 0.75
        score = 0.0
        for term in token_list(query, remove_stopwords=True):
            if term not in frequencies:
                continue
            frequency = frequencies[term]
            denominator = frequency + k1 * (1 - b + b * len(document) / max(1.0, self.avg_len))
            score += self.idf.get(term, 0.0) * (frequency * (k1 + 1)) / max(denominator, 1e-9)
        return score

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        scored = sorted(((self.score_node(query, node), node) for node in self.clauses), key=lambda item: item[0], reverse=True)
        return [evidence_from_node(node, score, [node.id], role="retrieved", retriever=self.name) for score, node in scored[:top_k]]


class CommunityGraphRetriever:
    """Concept-community retrieval inspired by community-summary GraphRAG.

    This is a transparent baseline, not the Microsoft GraphRAG implementation.
    It ranks concept communities and then selects representative clauses.
    """

    name = "Community GraphRAG"

    def __init__(self, graph: PropertyGraph):
        self.graph = graph

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        query_concepts = set(infer_concepts(query))
        concept_scores: dict[str, float] = {}
        for node in self.graph.nodes.values():
            if node.type != "Concept":
                continue
            score = token_similarity(query, node.label)
            if node.label in query_concepts:
                score += 0.8
            concept_scores[node.id] = score

        clause_scores: dict[str, float] = defaultdict(float)
        paths: dict[str, list[str]] = {}
        for concept_id, concept_score in concept_scores.items():
            for edge in self.graph.incoming.get(concept_id, []):
                if edge.type != "APPLIES_TO":
                    continue
                clause = self.graph.nodes[edge.source]
                lexical = token_similarity(query, f"{clause.label} {clause.properties.get('text', '')}")
                score = 0.6 * concept_score + 0.4 * lexical
                if score > clause_scores[clause.id]:
                    clause_scores[clause.id] = score
                    paths[clause.id] = [concept_id, clause.id]

        if not clause_scores:
            for score, clause in self.graph.ranked_clauses(query, top_k):
                clause_scores[clause.id] = score
                paths[clause.id] = [clause.id]

        ranked = sorted(clause_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:top_k]
        return [
            evidence_from_node(self.graph.nodes[clause_id], score, paths[clause_id], role="community", retriever=self.name)
            for clause_id, score in ranked
        ]


class PPRGraphRetriever:
    """Personalized-PageRank retrieval inspired by HippoRAG."""

    name = "PPR GraphRAG"

    def __init__(self, graph: PropertyGraph, damping: float = 0.85, iterations: int = 24):
        self.graph = graph
        self.damping = damping
        self.iterations = iterations

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        node_ids = sorted(self.graph.nodes)
        if not node_ids:
            return []
        seeds: dict[str, float] = {}
        for score, clause in self.graph.ranked_clauses(query, max(top_k, 6)):
            seeds[clause.id] = max(score, 1e-6)
        for concept in infer_concepts(query):
            concept_id = f"concept:{concept}"
            if concept_id in self.graph.nodes:
                seeds[concept_id] = max(seeds.get(concept_id, 0.0), 1.0)
        total = sum(seeds.values()) or 1.0
        personalization = {node_id: seeds.get(node_id, 0.0) / total for node_id in node_ids}
        rank = dict(personalization)

        adjacency: dict[str, list[str]] = {}
        for node_id in node_ids:
            adjacency[node_id] = [node.id for node in self.graph.neighbors(node_id)]

        for _ in range(self.iterations):
            updated = {node_id: (1.0 - self.damping) * personalization[node_id] for node_id in node_ids}
            for source in node_ids:
                neighbors = adjacency[source]
                if not neighbors:
                    continue
                share = self.damping * rank.get(source, 0.0) / len(neighbors)
                for target in neighbors:
                    updated[target] += share
            rank = updated

        clauses = [(rank.get(node.id, 0.0), node) for node in self.graph.nodes.values() if node.type == "Clause"]
        clauses.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        output: list[Evidence] = []
        seed_ids = list(seeds)
        for score, node in clauses[:top_k]:
            nearest = max(seed_ids, key=lambda seed: token_similarity(node.id, seed), default=node.id)
            path = self.graph.shortest_path(nearest, node.id) or [node.id]
            output.append(evidence_from_node(node, score, path, role="associative", retriever=self.name))
        return output


class SteinerGraphRetriever:
    """Connected-subgraph heuristic inspired by G-Retriever's PCST objective.

    It is intentionally dependency-free and must not be described as a faithful
    reproduction of the original trained G-Retriever model.
    """

    name = "Steiner GraphRAG"

    def __init__(self, graph: PropertyGraph):
        self.graph = graph

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        seeds = self.graph.ranked_clauses(query, max(3, min(top_k, 5)))
        if not seeds:
            return []
        selected_ids = {node.id for _, node in seeds}
        path_bonus: Counter[str] = Counter()
        seed_nodes = [node.id for _, node in seeds]
        for index, source in enumerate(seed_nodes):
            for target in seed_nodes[index + 1:]:
                path = self.graph.shortest_path(source, target, max_depth=7)
                for node_id in path:
                    if self.graph.nodes.get(node_id) and self.graph.nodes[node_id].type == "Clause":
                        selected_ids.add(node_id)
                        path_bonus[node_id] += 1

        seed_score = {node.id: score for score, node in seeds}
        ranked: list[tuple[float, Node, list[str]]] = []
        for clause_id in selected_ids:
            node = self.graph.nodes[clause_id]
            lexical = token_similarity(query, f"{node.label} {node.properties.get('text', '')}")
            score = seed_score.get(clause_id, 0.0) + 0.08 * path_bonus[clause_id] + 0.25 * lexical
            nearest_seed = max(seed_nodes, key=lambda seed: seed_score.get(seed, 0.0))
            path = self.graph.shortest_path(nearest_seed, clause_id, max_depth=7) or [clause_id]
            ranked.append((score, node, path))
        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        return [evidence_from_node(node, score, path, role="connected-subgraph", retriever=self.name) for score, node, path in ranked[:top_k]]


class VeriWeaveRetriever:
    """Retrieve support, counterevidence, and precedence evidence as one bundle."""

    name = "VeriWeave"

    def __init__(self, graph: PropertyGraph):
        self.graph = graph
        self.text = TextRetriever(graph)
        self.community = CommunityGraphRetriever(graph)
        self.ppr = PPRGraphRetriever(graph)

    def retrieve(self, query: str, top_k: int = 6) -> list[Evidence]:
        candidate_map: dict[str, Evidence] = {}
        for retriever, weight in ((self.text, 0.35), (self.community, 0.35), (self.ppr, 0.30)):
            for evidence in retriever.retrieve(query, max(top_k, 6)):
                existing = candidate_map.get(evidence.clause_id)
                weighted = evidence.score * weight
                if existing is None:
                    candidate_map[evidence.clause_id] = Evidence(**{**evidence.to_dict(), "score": weighted, "role": "support-candidate", "retriever": self.name})
                else:
                    candidate_map[evidence.clause_id] = Evidence(**{**existing.to_dict(), "score": existing.score + weighted})

        ranked_support = sorted(candidate_map.values(), key=lambda item: (item.score, item.clause_id), reverse=True)
        bundle: dict[str, Evidence] = {item.clause_id: item for item in ranked_support[:top_k]}

        # Deliberately retrieve counterevidence and superseding versions.
        for evidence in list(bundle.values()):
            for edge in self.graph.outgoing.get(evidence.clause_id, []) + self.graph.incoming.get(evidence.clause_id, []):
                if edge.type not in {"CONTRADICTS", "SUPPORTS"}:
                    continue
                other_id = edge.target if edge.source == evidence.clause_id else edge.source
                other = self.graph.nodes.get(other_id)
                if not other or other.type != "Clause":
                    continue
                role = "counterevidence" if edge.type == "CONTRADICTS" else "corroboration"
                score = evidence.score * (0.92 if role == "counterevidence" else 0.72)
                candidate = evidence_from_node(other, score, [evidence.clause_id, other.id], role=role, retriever=self.name)
                if candidate.clause_id not in bundle or candidate.score > bundle[candidate.clause_id].score:
                    bundle[candidate.clause_id] = candidate

            document = self.graph.document_for_clause(evidence.clause_id)
            if not document:
                continue
            for edge in self.graph.outgoing.get(document.id, []) + self.graph.incoming.get(document.id, []):
                if edge.type != "OVERRIDES":
                    continue
                other_doc_id = edge.target if edge.source == document.id else edge.source
                for incoming in self.graph.incoming.get(other_doc_id, []):
                    if incoming.type != "DERIVED_FROM":
                        continue
                    other = self.graph.nodes[incoming.source]
                    similarity = token_similarity(evidence.text, str(other.properties.get("text", "")))
                    if similarity < 0.08:
                        continue
                    role = "precedence" if other.properties.get("is_current") else "superseded"
                    score = evidence.score + (0.25 if role == "precedence" else -0.05)
                    candidate = evidence_from_node(other, score, [evidence.clause_id, document.id, other_doc_id, other.id], role=role, retriever=self.name)
                    if candidate.clause_id not in bundle or candidate.score > bundle[candidate.clause_id].score:
                        bundle[candidate.clause_id] = candidate

        priority = {"counterevidence": 4, "precedence": 3, "support-candidate": 2, "corroboration": 1, "superseded": 0}
        output = sorted(bundle.values(), key=lambda item: (priority.get(item.role, 0), item.score, item.clause_id), reverse=True)
        # Permit extra bundle items because counterevidence should not displace support.
        return output[: max(top_k + 4, top_k)]


class AuditRetriever:
    """Common post-hoc evidence retrieval used to evaluate every method equally."""

    name = "Independent Audit"

    def __init__(self, graph: PropertyGraph):
        self.veriweave = VeriWeaveRetriever(graph)

    def retrieve(self, question: str, answer: str, top_k: int = 10) -> list[Evidence]:
        query = f"{question}\n{answer}"
        return [Evidence(**{**item.to_dict(), "retriever": self.name}) for item in self.veriweave.retrieve(query, top_k)]
