import unittest

from veriweave.boltzmann_policy_attention import attend_policy_candidates
from veriweave.vita import analyze_vita_decision_space
from veriweave.graph import Edge, Node, PropertyGraph
from veriweave.models import AtomicClaim
from veriweave.retrieval import Evidence


def evidence(clause_id, citation, text, modality, concepts, *, current=True, rank=2, score=0.6, source=None):
    return Evidence(
        clause_id=clause_id,
        citation_id=citation,
        text=text,
        score=score,
        source=source or citation.split("#")[0],
        version="v2" if current else "v1",
        concepts=concepts,
        graph_path=[clause_id],
        role="test",
        retriever="test",
        modality=modality,
        is_current=current,
        authority_rank=rank,
    )


def graph_fixture():
    graph = PropertyGraph()
    for doc_id, version, rank in (("doc:seed", "v1", 1), ("doc:old", "v1", 1), ("doc:new", "v2", 3), ("doc:other", "v1", 1)):
        graph.add_node(Node(doc_id, "Document", doc_id, {"version": version, "authority_rank": rank}))
    graph.add_node(Node("concept:human-review", "Concept", "human-review", {}))
    clauses = [
        ("clause:seed", "The AI system may decide without human review.", "seed.md#rule@v1", "permission", True, 1, "doc:seed"),
        ("clause:old", "The AI system may decide without human review during a pilot.", "policy.md#rule@v1", "permission", False, 1, "doc:old"),
        ("clause:new", "The AI system must not decide without human review.", "policy.md#rule@v2", "prohibition", True, 3, "doc:new"),
        ("clause:other", "The system owner records routine maintenance.", "other.md#maintenance@v1", "guidance", True, 1, "doc:other"),
    ]
    for clause_id, text, citation, modality, current, rank, doc_id in clauses:
        graph.add_node(Node(clause_id, "Clause", clause_id, {
            "text": text,
            "citation_id": citation,
            "source": citation.split("#")[0],
            "version": "v2" if current and clause_id == "clause:new" else "v1",
            "concepts": ["human-review"] if clause_id != "clause:other" else ["auditability"],
            "modality": modality,
            "is_current": current,
            "authority_rank": rank,
        }))
        graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
    graph.add_edge(Edge("clause:seed", "concept:human-review", "APPLIES_TO"))
    graph.add_edge(Edge("clause:old", "concept:human-review", "APPLIES_TO"))
    graph.add_edge(Edge("clause:new", "concept:human-review", "APPLIES_TO"))
    graph.add_edge(Edge("clause:old", "clause:new", "CONTRADICTS"))
    graph.add_edge(Edge("clause:new", "clause:old", "CONTRADICTS"))
    graph.add_edge(Edge("doc:new", "doc:old", "OVERRIDES"))
    return graph


class BoltzmannPolicyAttentionTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.seed = [evidence(
            "clause:seed", "seed.md#rule@v1",
            "The AI system may decide without human review.",
            "permission", ["human-review"], current=True, rank=1, score=0.9,
        )]
        self.candidates = [
            evidence("clause:old", "policy.md#rule@v1", "The AI system may decide without human review during a pilot.", "permission", ["human-review"], current=False, rank=1, score=0.55, source="policy.md"),
            evidence("clause:new", "policy.md#rule@v2", "The AI system must not decide without human review.", "prohibition", ["human-review"], current=True, rank=3, score=0.58, source="policy.md"),
            evidence("clause:other", "other.md#maintenance@v1", "The system owner records routine maintenance.", "guidance", ["auditability"], current=True, rank=1, score=0.60),
        ]
        self.claims = [AtomicClaim(
            id="claim:decision",
            text="The AI system is allowed to decide without human review.",
            kind="decision",
            decisive=True,
        )]

    def test_distribution_is_normalized_and_certified(self):
        selected, certificate = attend_policy_candidates(
            "Can the AI decide without human review?",
            self.claims, self.seed, self.candidates, self.graph, "allowed",
            max_coalition_size=2, selection_budget=3, attention_mass=0.95,
        )
        self.assertTrue(selected)
        self.assertAlmostEqual(sum(row["probability"] for row in certificate.top_coalitions), 1.0, places=5)
        self.assertGreater(certificate.effective_sample_size, 1.0)
        self.assertLessEqual(certificate.residual_probability_mass, 1.0)

    def test_lower_temperature_concentrates_attention(self):
        _, cold = attend_policy_candidates(
            "Can the AI decide without human review?",
            self.claims, self.seed, self.candidates, self.graph, "allowed",
            base_temperature=0.18, minimum_temperature=0.05, maximum_temperature=2.0,
            max_coalition_size=2, selection_budget=3,
        )
        _, hot = attend_policy_candidates(
            "Can the AI decide without human review?",
            self.claims, self.seed, self.candidates, self.graph, "allowed",
            base_temperature=1.8, minimum_temperature=0.05, maximum_temperature=3.0,
            max_coalition_size=2, selection_budget=3,
        )
        self.assertLess(cold.normalized_entropy, hot.normalized_entropy)

    def test_pairwise_coupling_surfaces_version_conflict(self):
        _, certificate = attend_policy_candidates(
            "Can the AI decide without human review?",
            self.claims, self.seed, self.candidates, self.graph, "allowed",
            base_temperature=0.35, max_coalition_size=2, selection_budget=2,
        )
        strongest = certificate.top_coalitions[0]["clause_ids"]
        self.assertIn("clause:new", strongest)
        self.assertTrue(any(
            row["left_clause_id"] in {"clause:old", "clause:new"}
            and row["right_clause_id"] in {"clause:old", "clause:new"}
            and row["coupling"] > 0
            for row in certificate.pairwise_couplings
        ))

    def test_vita_emits_boltzmann_certificate(self):
        vita, *_middle, attention = analyze_vita_decision_space(
            "Can the AI decide without human review?",
            self.claims, self.seed, self.graph, "allowed",
            max_candidates=3, max_coalition_size=2, max_rounds=1,
            use_boltzmann_attention=True, boltzmann_max_candidates=3,
        )
        self.assertIsNotNone(attention)
        self.assertIn("clause:new", attention.selected_clause_ids)
        self.assertIn("needs_review", vita.reachable_decisions)


if __name__ == "__main__":
    unittest.main()
