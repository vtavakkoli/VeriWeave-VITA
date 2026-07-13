import unittest
from pathlib import Path

from claimgraph.data_loader import load_policies
from claimgraph.graph import build_property_graph


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))

    def test_graph_contains_typed_nodes(self):
        types = {node.type for node in self.graph.nodes.values()}
        self.assertTrue({"Document", "Clause", "Concept", "Version", "Authority"}.issubset(types))

    def test_version_precedence_exists(self):
        self.assertTrue(any(edge.type == "OVERRIDES" for edge in self.graph.edges))

    def test_clause_provenance_exists(self):
        clause = next(node for node in self.graph.nodes.values() if node.type == "Clause")
        self.assertTrue(any(edge.type == "DERIVED_FROM" for edge in self.graph.outgoing[clause.id]))


if __name__ == "__main__":
    unittest.main()
