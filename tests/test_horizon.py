import unittest

from veriweave.graph import Edge, Node, PropertyGraph
from veriweave.horizon import analyze_evidence_horizon
from veriweave.models import AtomicClaim
from veriweave.retrieval import Evidence


class EvidenceHorizonTests(unittest.TestCase):
    def setUp(self):
        graph = PropertyGraph()
        graph.add_node(Node("authority:1", "Authority", "Department", {"rank": 1}))
        graph.add_node(Node("authority:2", "Authority", "Regulator", {"rank": 2}))
        graph.add_node(Node("document:old", "Document", "Old policy", {"version": "v1", "authority_rank": 1}))
        graph.add_node(Node("document:new", "Document", "New policy", {"version": "v2", "authority_rank": 2}))
        graph.add_node(Node("concept:human-review", "Concept", "human-review", {}))
        graph.add_node(Node("clause:permission", "Clause", "Permission", {
            "text": "The AI system may make the decision without human review.",
            "citation_id": "old.md#permission@v1",
            "source": "old.md",
            "version": "v1",
            "concepts": ["human-review"],
            "modality": "permission",
            "is_current": False,
            "authority_rank": 1,
        }))
        graph.add_node(Node("clause:prohibition", "Clause", "Prohibition", {
            "text": "The AI system must not make the decision without human review.",
            "citation_id": "new.md#prohibition@v2",
            "source": "new.md",
            "version": "v2",
            "concepts": ["human-review"],
            "modality": "prohibition",
            "is_current": True,
            "authority_rank": 2,
        }))
        graph.add_edge(Edge("clause:permission", "document:old", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:prohibition", "document:new", "DERIVED_FROM"))
        graph.add_edge(Edge("document:old", "authority:1", "GOVERNED_BY"))
        graph.add_edge(Edge("document:new", "authority:2", "GOVERNED_BY"))
        graph.add_edge(Edge("clause:permission", "concept:human-review", "APPLIES_TO"))
        graph.add_edge(Edge("clause:prohibition", "concept:human-review", "APPLIES_TO"))
        graph.add_edge(Edge("clause:permission", "clause:prohibition", "CONTRADICTS"))
        graph.add_edge(Edge("document:new", "document:old", "OVERRIDES"))
        self.graph = graph
        self.initial = [Evidence(
            clause_id="clause:permission",
            citation_id="old.md#permission@v1",
            text="The AI system may make the decision without human review.",
            score=0.9,
            source="old.md",
            version="v1",
            concepts=["human-review"],
            graph_path=["clause:permission"],
            role="retrieved",
            retriever="test",
            modality="permission",
            is_current=False,
            authority_rank=1,
        )]
        self.claims = [AtomicClaim(
            id="claim:decision",
            text="The AI system is allowed to make the decision without human review.",
            kind="decision",
            polarity="positive",
            decisive=True,
        )]

    def test_horizon_finds_unseen_higher_precedence_counterevidence(self):
        certificate, expanded, _, _ = analyze_evidence_horizon(
            "Can the AI system decide without human review?",
            self.claims,
            self.initial,
            self.graph,
            "allowed",
        )
        self.assertTrue(certificate.decision_changing_blind_spots)
        self.assertEqual(certificate.expanded_decision, "needs_review")
        self.assertIn("new.md#prohibition@v2", {item.citation_id for item in expanded})
        self.assertTrue(certificate.requires_review)

    def test_cut_is_not_claimed_when_no_supported_decisive_claim_remains(self):
        certificate, _, _, _ = analyze_evidence_horizon(
            "Can the AI system decide without human review?",
            self.claims,
            self.initial,
            self.graph,
            "allowed",
        )
        self.assertIsNone(certificate.minimal_evidence_cut)
        self.assertEqual(certificate.evidence_cut_robustness, 0.0)


if __name__ == "__main__":
    unittest.main()
