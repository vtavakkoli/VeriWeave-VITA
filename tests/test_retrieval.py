import unittest
from pathlib import Path

from veriweave.data_loader import load_policies
from veriweave.graph import build_property_graph
from veriweave.retrieval import (
    CommunityGraphRetriever,
    PPRGraphRetriever,
    SteinerGraphRetriever,
    TextRetriever,
    VeriWeaveRetriever,
)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))

    def test_all_baselines_return_paths(self):
        query = "Which current version applies when policies conflict?"
        for retriever in (
            CommunityGraphRetriever(self.graph),
            PPRGraphRetriever(self.graph),
            SteinerGraphRetriever(self.graph),
        ):
            evidence = retriever.retrieve(query, 4)
            self.assertTrue(evidence, retriever.name)
            self.assertTrue(all(item.graph_path for item in evidence), retriever.name)

    def test_text_retrieval_finds_human_review(self):
        evidence = TextRetriever(self.graph).retrieve("human review for an adverse decision", 3)
        self.assertTrue(any("review" in item.text.lower() for item in evidence))

    def test_community_retrieval_handles_zero_overlap_candidates(self):
        # Regression for T015: zero-scored concept candidates used to be
        # inserted by defaultdict without a corresponding graph path.
        query = "What accountability information should an AI-assisted recommendation identify?"
        evidence = CommunityGraphRetriever(self.graph).retrieve(query, 10)
        self.assertTrue(evidence)
        self.assertTrue(all(item.graph_path for item in evidence))

    def test_veriweave_retrieves_multiple_evidence_roles(self):
        evidence = VeriWeaveRetriever(self.graph).retrieve("current rule when policy versions conflict", 6)
        roles = {item.role for item in evidence}
        self.assertIn("support-candidate", roles)
        self.assertTrue(roles & {"precedence", "superseded", "counterevidence", "corroboration"})


if __name__ == "__main__":
    unittest.main()
