import unittest

from veriweave.data_loader import BenchmarkTask
from veriweave.evaluators import evaluate


class EvaluatorMetricTests(unittest.TestCase):
    def test_one_winning_citation_covers_supported_claim(self):
        task = BenchmarkTask(
            id="T", task_type="test", difficulty="easy", question="Q", expected_answer="A",
            expected_decision="allowed", expected_human_review=False, expected_policy_refs=[]
        )
        trace = {
            "answer": "A supported answer.", "decision": "allowed", "human_review_required": False,
            "citations": ["a#1@v1"],
            "audit_claims": [{"id": "c1"}],
            "audit_validations": [{
                "claim_id": "c1", "status": "supported", "provenance_complete": True,
                "temporal_validity_score": 1.0, "applicability_score": 0.3,
                "evidence_ids": ["a#1@v1", "b#2@v1"],
                "winning_evidence_ids": ["a#1@v1", "b#2@v1"],
            }],
            "audit_evidence": [
                {"citation_id": "a#1@v1"}, {"citation_id": "b#2@v1"}
            ],
            "evidence": [{
                "citation_id": "a#1@v1", "source": "a", "version": "v1", "graph_path": ["c"]
            }],
            "audit_horizon": {"horizon_stability": 1.0, "evidence_cut_robustness": 1.0},
            "audit_vita": {"invariant_under_tested_closure": True, "closure_converged": True, "residual_risk_mass": 0.0},
            "audit_argumentation": {"conflict_free": True},
            "audit_temporal_drift": {"stable_across_versions": True},
            "audit_boltzmann_policy_attention": {"selected_attention_mass": 1.0},
        }
        result = evaluate(task, trace)
        self.assertEqual(result["citation_coverage"], 1.0)
        self.assertEqual(result["graph_path_provenance"], 1.0)
        self.assertEqual(result["source_version_provenance"], 1.0)

    def test_no_supported_claim_has_zero_cut_robustness(self):
        task = BenchmarkTask(
            id="T", task_type="test", difficulty="easy", question="Q", expected_answer="A",
            expected_decision="needs_review", expected_human_review=True, expected_policy_refs=[]
        )
        trace = {
            "answer": "Unverified.", "decision": "needs_review", "human_review_required": True,
            "citations": [], "audit_claims": [{"id": "c1"}],
            "audit_validations": [{
                "claim_id": "c1", "status": "unresolved", "provenance_complete": False,
                "temporal_validity_score": 1.0, "applicability_score": 0.0,
                "evidence_ids": [], "winning_evidence_ids": [],
            }],
            "audit_evidence": [], "evidence": [],
            "audit_horizon": {"horizon_stability": 1.0, "evidence_cut_robustness": 1.0},
        }
        result = evaluate(task, trace)
        self.assertEqual(result["evidence_cut_robustness"], 0.0)


if __name__ == "__main__":
    unittest.main()
