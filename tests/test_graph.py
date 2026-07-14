import unittest
from pathlib import Path

from veriweave.data_loader import load_policies
from veriweave.graph import build_property_graph


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

    def test_shortest_path_is_deterministic(self):
        clauses = sorted(node.id for node in self.graph.nodes.values() if node.type == "Clause")
        self.assertEqual(self.graph.shortest_path(clauses[0], clauses[0]), [clauses[0]])


if __name__ == "__main__":
    unittest.main()
