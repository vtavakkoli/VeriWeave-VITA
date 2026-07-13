import unittest
from pathlib import Path

from claimgraph.data_loader import load_policies
from claimgraph.graph import build_property_graph
from claimgraph.retrieval import GraphRetriever, TextRetriever


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))

    def test_graph_retrieval_has_paths(self):
        evidence = GraphRetriever(self.graph).retrieve("Which current version applies when policies conflict?", 4)
        self.assertTrue(evidence)
        self.assertTrue(all(item.graph_path for item in evidence))

    def test_text_retrieval_finds_human_review(self):
        evidence = TextRetriever(self.graph).retrieve("human review for an adverse decision", 3)
        self.assertTrue(any("review" in item.text.lower() for item in evidence))


if __name__ == "__main__":
    unittest.main()
