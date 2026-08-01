import unittest

from veriweave.envelope import build_verification_envelope
from veriweave.graph import Edge, Node, PropertyGraph
from veriweave.models import AtomicClaim, CandidateResponse, ClaimValidation
from veriweave.precedence import resolve_precedence
from veriweave.provenance_robust_selection import select_provenance_robust_candidates
from veriweave.retrieval import Evidence
from veriweave.utils import extract_citation_ids, strip_citations
from veriweave.vita import _vita_decision, analyze_vita_decision_space


def evidence(
    clause_id: str,
    citation: str,
    text: str,
    modality: str = "guidance",
    concepts: list[str] | None = None,
    *,
    current: bool = True,
    authority: int = 2,
    score: float = 0.8,
    source: str | None = None,
    version: str | None = None,
) -> Evidence:
    return Evidence(
        clause_id=clause_id,
        citation_id=citation,
        text=text,
        score=score,
        source=source or citation.split("#")[0],
        version=version or ("v2" if current else "v1"),
        concepts=concepts or ["policy"],
        graph_path=[clause_id],
        role="test",
        retriever="test",
        modality=modality,
        is_current=current,
        authority_rank=authority,
    )


def add_evidence_node(graph: PropertyGraph, item: Evidence) -> None:
    graph.add_node(Node(
        item.clause_id,
        "Clause",
        item.clause_id,
        {
            "text": item.text,
            "citation_id": item.citation_id,
            "source": item.source,
            "version": item.version,
            "concepts": item.concepts,
            "modality": item.modality,
            "is_current": item.is_current,
            "authority_rank": item.authority_rank,
        },
    ))


def supported_validation(claim: AtomicClaim, citation: str) -> ClaimValidation:
    return ClaimValidation(
        claim_id=claim.id,
        claim=claim.text,
        status="supported",
        support_score=1.0,
        contradiction_score=0.0,
        applicability_score=1.0,
        temporal_validity_score=1.0,
        provenance_complete=True,
        evidence_ids=[citation],
        counterevidence_ids=[],
        winning_evidence_ids=[citation],
        reasons=[],
    )


class TraceRegressionTests(unittest.TestCase):
    def test_inline_citations_do_not_pollute_semantic_text(self):
        text = "The pilot requires security review. [pilot-policy-v2.md#controls@v2]"
        self.assertEqual(strip_citations(text), "The pilot requires security review.")
        self.assertEqual(extract_citation_ids(text), ["pilot-policy-v2.md#controls@v2"])

    def test_human_review_condition_is_not_a_categorical_ban(self):
        item = evidence(
            "clause:review",
            "high-risk.md#review@v2",
            "The system must not make benefits decisions without trained human review.",
            "prohibition",
            ["human-review", "high-risk-ai"],
        )
        claim = AtomicClaim(
            "claim:decision",
            item.text,
            kind="decision",
            decisive=True,
        )
        self.assertEqual(
            _vita_decision(
                "not_allowed",
                [claim],
                [item],
                [supported_validation(claim, item.citation_id)],
                [],
            ),
            "needs_review",
        )

    def test_unconditional_prohibition_remains_not_allowed(self):
        item = evidence(
            "clause:ban",
            "biometrics.md#ban@v2",
            "Real-time biometric categorization is prohibited.",
            "prohibition",
            ["biometrics"],
        )
        claim = AtomicClaim("claim:ban", item.text, kind="decision", decisive=True)
        self.assertEqual(
            _vita_decision(
                "not_allowed",
                [claim],
                [item],
                [supported_validation(claim, item.citation_id)],
                [],
            ),
            "not_allowed",
        )

    def test_unrelated_sources_are_not_false_version_conflicts(self):
        graph = PropertyGraph()
        left = evidence(
            "clause:left",
            "audit-policy.md#scope@v1",
            "Audit records contain a timestamp and owner.",
            source="audit-policy.md",
            version="v1",
        )
        right = evidence(
            "clause:right",
            "security-standard.md#scope@v2",
            "Security records contain a timestamp and owner.",
            source="security-standard.md",
            version="v2",
        )
        self.assertEqual(resolve_precedence([left, right], graph), [])

    def test_pro_selects_current_decisive_rule_before_stale_diversity(self):
        graph = PropertyGraph()
        seed = evidence(
            "clause:seed",
            "scope.md#scope@v2",
            "This policy covers automated benefits decisions.",
            concepts=["benefits", "human-review"],
        )
        current = evidence(
            "clause:current",
            "high-risk.md#review@v2",
            "Benefits decisions require trained human review.",
            "obligation",
            ["benefits", "human-review"],
            authority=3,
            score=0.65,
        )
        stale = evidence(
            "clause:stale",
            "legacy.md#automation@v1",
            "Benefits decisions may proceed automatically.",
            "permission",
            ["benefits", "human-review"],
            current=False,
            authority=1,
            score=0.98,
        )
        distractor = evidence(
            "clause:distractor",
            "logging.md#retention@v2",
            "Pilot logs are retained for thirty days.",
            "guidance",
            ["auditability"],
            score=0.99,
        )
        add_evidence_node(graph, current)
        add_evidence_node(graph, stale)
        graph.add_edge(Edge("clause:current", "clause:stale", "CONTRADICTS"))
        claim = AtomicClaim(
            "claim:review",
            "Benefits decisions require trained human review.",
            kind="decision",
            decisive=True,
        )
        selected, certificate = select_provenance_robust_candidates(
            "May benefits decisions run without human review?",
            [claim],
            [seed],
            [stale, distractor, current],
            graph,
            selection_budget=1,
        )
        self.assertEqual([item.clause_id for item in selected], ["clause:current"])
        self.assertEqual(certificate.current_evidence_rate, 1.0)

    def test_conflict_question_routes_to_review(self):
        graph = PropertyGraph()
        item = evidence(
            "clause:current",
            "high-risk.md#review@v2",
            "High-risk benefits decisions require trained human oversight.",
            "obligation",
            ["human-review", "high-risk-ai"],
        )
        candidate = CandidateResponse(
            answer=f"The high-risk oversight rule applies. [{item.citation_id}]",
            decision="conditional",
            citations=[item.citation_id],
            human_review_required=False,
        )
        envelope = build_verification_envelope(
            candidate,
            [item],
            graph,
            question="Optional departmental guidance conflicts with high-risk oversight. Which applies?",
            task_type="conflict_and_version_reasoning",
            enable_horizon=False,
        )
        self.assertEqual(envelope.gated_decision, "needs_review")
        self.assertTrue(envelope.human_review_required)

    def test_nondecisive_removed_claim_is_advisory_not_blocking(self):
        graph = PropertyGraph()
        item = evidence(
            "clause:audit",
            "audit.md#scope@v2",
            "The policy covers audit records.",
            "guidance",
            ["auditability"],
        )
        candidate = CandidateResponse(
            answer=f"The policy covers audit records. [{item.citation_id}] The moon is blue.",
            decision="allowed",
            citations=[item.citation_id],
            human_review_required=False,
        )
        envelope = build_verification_envelope(
            candidate,
            [item],
            graph,
            question="Does the policy cover audit records?",
            enable_horizon=False,
        )
        self.assertEqual(envelope.gated_decision, "allowed")
        self.assertFalse(envelope.human_review_required)
        self.assertTrue(envelope.advisory_reasons)
        self.assertFalse(envelope.review_reasons)

    def test_residual_search_mass_is_reported_without_forcing_review(self):
        graph = PropertyGraph()
        base = evidence(
            "clause:base",
            "audit.md#scope@v2",
            "The policy covers audit records.",
            "guidance",
            ["auditability"],
        )
        add_evidence_node(graph, base)
        for index in range(5):
            add_evidence_node(graph, evidence(
                f"clause:extra:{index}",
                f"audit-{index}.md#detail@v2",
                f"Audit records include detail {index}.",
                "guidance",
                ["auditability"],
            ))
        claim = AtomicClaim(
            "claim:scope",
            "The policy covers audit records.",
            kind="assertion",
            decisive=False,
        )
        certificate, *_ = analyze_vita_decision_space(
            "Does the policy cover audit records?",
            [claim],
            [base],
            graph,
            "allowed",
            max_candidates=1,
            max_coalition_size=1,
            max_rounds=1,
        )
        self.assertFalse(certificate.requires_review)
        self.assertTrue(certificate.advisory_reasons)


if __name__ == "__main__":
    unittest.main()
