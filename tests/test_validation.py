import unittest
from pathlib import Path

from veriweave.data_loader import load_policies
from veriweave.graph import build_property_graph
from veriweave.models import AtomicClaim, CandidateResponse
from veriweave.precedence import resolve_precedence
from veriweave.retrieval import VeriWeaveRetriever
from veriweave.validators import validate_claim
from veriweave.envelope import build_verification_envelope


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.graph = build_property_graph(load_policies(root / "data"))
        cls.retriever = VeriWeaveRetriever(cls.graph)

    def test_supported_claim_has_provenance(self):
        evidence = self.retriever.retrieve("external cloud requires security and data protection review", 6)
        resolutions = resolve_precedence(evidence, self.graph)
        result = validate_claim(
            AtomicClaim("c1", "External cloud processing requires security and data-protection review."),
            evidence,
            self.graph,
            resolutions,
        )
        self.assertTrue(result.provenance_complete)
        self.assertGreater(result.support_score, 0.15)

    def test_envelope_does_not_invent_citations(self):
        evidence = self.retriever.retrieve("audit trail requirements", 6)
        candidate = CandidateResponse(
            answer="An audit trail is required.",
            decision="conditional",
            citations=[],
            human_review_required=False,
        )
        envelope = build_verification_envelope(candidate, evidence, self.graph)
        self.assertEqual(envelope.accepted_citations, [])

    def test_unverified_decision_is_gated(self):
        evidence = self.retriever.retrieve("employee adverse decision human review", 6)
        candidate = CandidateResponse(
            answer="The system may make the final adverse employment decision without review.",
            decision="allowed",
            citations=[],
            human_review_required=False,
        )
        envelope = build_verification_envelope(candidate, evidence, self.graph)
        self.assertEqual(envelope.gated_decision, "needs_review")
        self.assertTrue(envelope.human_review_required)


if __name__ == "__main__":
    unittest.main()
