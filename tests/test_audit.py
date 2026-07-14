import unittest
from pathlib import Path

from veriweave.audit import attach_independent_audit
from veriweave.data_loader import load_policies
from veriweave.graph import build_property_graph
from veriweave.retrieval import AuditRetriever


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))
        cls.audit = AuditRetriever(cls.graph)

    def test_same_audit_fields_can_be_attached_to_any_method(self):
        trace = {
            "method": "Direct LLM",
            "question": "What must an audit trail contain?",
            "answer": "An audit trail must contain a timestamp and the final decision.",
        }
        attach_independent_audit(trace, self.graph, self.audit)
        self.assertTrue(trace["audit_claims"])
        self.assertTrue(trace["audit_evidence"])
        self.assertEqual(len(trace["audit_claims"]), len(trace["audit_validations"]))


if __name__ == "__main__":
    unittest.main()
