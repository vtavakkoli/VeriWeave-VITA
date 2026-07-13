import unittest
from pathlib import Path

from claimgraph.claims import AtomicClaim
from claimgraph.data_loader import load_policies
from claimgraph.graph import build_property_graph
from claimgraph.retrieval import GraphRetriever
from claimgraph.validators import validate_claim


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))
        cls.retriever = GraphRetriever(cls.graph)

    def test_supported_claim_has_provenance(self):
        evidence = self.retriever.retrieve("external cloud requires security and data protection review", 6)
        result = validate_claim(AtomicClaim("c1", "External cloud processing requires security and data-protection review."), evidence, self.graph)
        self.assertTrue(result.provenance_complete)
        self.assertGreater(result.support_score, 0.15)

    def test_version_claim_requires_current_evidence(self):
        evidence = self.retriever.retrieve("current pilot policy version", 6)
        result = validate_claim(AtomicClaim("c2", "The current version requires a named owner."), evidence, self.graph)
        self.assertIn(result.temporal_validity_score, {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
