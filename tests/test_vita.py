import unittest

from veriweave.argumentation import build_argumentation_certificate
from veriweave.vita import analyze_vita_decision_space
from veriweave.graph import Edge, Node, PropertyGraph
from veriweave.models import AtomicClaim
from veriweave.precedence import resolve_precedence
from veriweave.retrieval import Evidence
from veriweave.temporal import build_temporal_drift_certificate


def evidence(clause_id, citation, text, modality, concepts, current=True, rank=2, score=0.8):
    return Evidence(
        clause_id=clause_id,
        citation_id=citation,
        text=text,
        score=score,
        source=citation.split("#")[0],
        version="v2" if current else "v1",
        concepts=concepts,
        graph_path=[clause_id],
        role="test",
        retriever="test",
        modality=modality,
        is_current=current,
        authority_rank=rank,
    )


class VITATests(unittest.TestCase):
    def test_coalitional_closure_finds_synergy_missed_by_singletons(self):
        graph = PropertyGraph()
        graph.add_node(Node("doc:base", "Document", "Base Policy", {"version": "v1", "authority_rank": 2}))
        graph.add_node(Node("doc:a", "Document", "Log Policy A", {"version": "v1", "authority_rank": 2}))
        graph.add_node(Node("doc:b", "Document", "Log Policy B", {"version": "v1", "authority_rank": 2}))
        graph.add_node(Node("concept:human-review", "Concept", "human-review", {}))
        graph.add_node(Node("concept:auditability", "Concept", "auditability", {}))
        graph.add_node(Node("clause:base", "Clause", "Decision", {
            "text": "The AI system may make the decision without human review.",
            "citation_id": "base.md#decision@v1",
            "source": "base.md", "version": "v1", "concepts": ["human-review", "auditability"],
            "modality": "permission", "is_current": True, "authority_rank": 2,
        }))
        graph.add_node(Node("clause:a", "Clause", "Log retention", {
            "text": "Audit copies are retained for the pilot.",
            "citation_id": "a.md#logs@v1",
            "source": "a.md", "version": "v1", "concepts": ["auditability"],
            "modality": "guidance", "is_current": True, "authority_rank": 2,
        }))
        graph.add_node(Node("clause:b", "Clause", "Log deletion", {
            "text": "Audit copies are deleted after the pilot.",
            "citation_id": "b.md#logs@v1",
            "source": "b.md", "version": "v1", "concepts": ["auditability"],
            "modality": "guidance", "is_current": True, "authority_rank": 2,
        }))
        for clause_id, doc_id in (("clause:base", "doc:base"), ("clause:a", "doc:a"), ("clause:b", "doc:b")):
            graph.add_edge(Edge(clause_id, doc_id, "DERIVED_FROM"))
        graph.add_edge(Edge("clause:base", "concept:human-review", "APPLIES_TO"))
        graph.add_edge(Edge("clause:base", "concept:auditability", "APPLIES_TO"))
        graph.add_edge(Edge("clause:a", "concept:auditability", "APPLIES_TO"))
        graph.add_edge(Edge("clause:b", "concept:auditability", "APPLIES_TO"))
        graph.add_edge(Edge("clause:a", "clause:b", "CONTRADICTS"))
        graph.add_edge(Edge("clause:b", "clause:a", "CONTRADICTS"))

        initial = [evidence(
            "clause:base", "base.md#decision@v1",
            "The AI system may make the decision without human review.",
            "permission", ["human-review", "auditability"], score=0.95,
        )]
        claims = [AtomicClaim(
            id="claim:decision",
            text="The AI system is allowed to make the decision without human review.",
            kind="decision", decisive=True,
        )]
        certificate, *_ = analyze_vita_decision_space(
            "Can the AI decide without human review?",
            claims,
            initial,
            graph,
            "allowed",
            max_candidates=4,
            max_coalition_size=2,
            max_rounds=1,
        )
        self.assertTrue(certificate.synergistic_blind_spots)
        self.assertIn("needs_review", certificate.reachable_decisions)
        self.assertGreater(certificate.decision_space_width, 0)


    def test_two_sided_vita_can_expose_overconservative_decision(self):
        graph = PropertyGraph()
        graph.add_node(Node("doc:scope", "Document", "Cloud Scope", {"version": "v1", "authority_rank": 2}))
        graph.add_node(Node("doc:permission", "Document", "Cloud Permission", {"version": "v1", "authority_rank": 2}))
        graph.add_node(Node("concept:external-service", "Concept", "external-service", {}))
        graph.add_node(Node("clause:scope", "Clause", "Scope", {
            "text": "This policy governs the use of external cloud services.",
            "citation_id": "scope.md#scope@v1", "source": "scope.md", "version": "v1",
            "concepts": ["external-service"], "modality": "guidance", "is_current": True, "authority_rank": 2,
        }))
        graph.add_node(Node("clause:permission", "Clause", "Permission", {
            "text": "External cloud services may be used for approved pilot workloads.",
            "citation_id": "permission.md#cloud@v1", "source": "permission.md", "version": "v1",
            "concepts": ["external-service"], "modality": "permission", "is_current": True, "authority_rank": 2,
        }))
        graph.add_edge(Edge("clause:scope", "doc:scope", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:permission", "doc:permission", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:scope", "concept:external-service", "APPLIES_TO"))
        graph.add_edge(Edge("clause:permission", "concept:external-service", "APPLIES_TO"))
        initial = [evidence(
            "clause:scope", "scope.md#scope@v1",
            "This policy governs the use of external cloud services.",
            "guidance", ["external-service"], score=0.95,
        )]
        claims = [AtomicClaim(
            id="claim:scope",
            text="The policy addresses external cloud services.",
            kind="assertion", decisive=False,
        )]
        certificate, *_ = analyze_vita_decision_space(
            "Does the policy address external cloud services?", claims, initial, graph, "not_allowed",
            max_candidates=3, max_coalition_size=1, max_rounds=1,
        )
        self.assertEqual(certificate.most_permissive_decision, "allowed")
        self.assertEqual(certificate.most_restrictive_decision, "not_allowed")
        self.assertGreater(certificate.decision_space_width, 0)

    def test_argumentation_uses_precedence_to_defeat_old_rule(self):
        graph = PropertyGraph()
        graph.add_node(Node("doc:old", "Document", "Decision Policy v1", {"version": "v1", "authority_rank": 1}))
        graph.add_node(Node("doc:new", "Document", "Decision Policy v2", {"version": "v2", "authority_rank": 2}))
        graph.add_node(Node("clause:old", "Clause", "Human review", {}))
        graph.add_node(Node("clause:new", "Clause", "Human review", {}))
        graph.add_edge(Edge("clause:old", "doc:old", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:new", "doc:new", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:old", "clause:new", "CONTRADICTS"))
        graph.add_edge(Edge("doc:new", "doc:old", "OVERRIDES"))
        old = evidence("clause:old", "old.md#review@v1", "The system may decide without review.", "permission", ["human-review"], current=False, rank=1)
        new = evidence("clause:new", "new.md#review@v2", "The system must not decide without review.", "prohibition", ["human-review"], current=True, rank=2)
        resolutions = resolve_precedence([old, new], graph)
        cert = build_argumentation_certificate([old, new], graph, resolutions)
        self.assertIn("clause:new", cert.accepted_argument_ids)
        self.assertIn("clause:old", cert.rejected_argument_ids)
        self.assertTrue(cert.conflict_free)

    def test_temporal_replay_detects_more_restrictive_drift(self):
        graph = PropertyGraph()
        graph.add_node(Node("doc:old", "Document", "Automated Decision Policy v1", {"version": "v1", "authority_rank": 1}))
        graph.add_node(Node("doc:new", "Document", "Automated Decision Policy v2", {"version": "v2", "authority_rank": 2}))
        graph.add_node(Node("clause:old", "Clause", "Human review", {
            "text": "The AI system may decide without human review.",
            "citation_id": "old.md#review@v1", "source": "old.md", "version": "v1",
            "concepts": ["human-review"], "modality": "permission", "is_current": False, "authority_rank": 1,
        }))
        graph.add_node(Node("clause:new", "Clause", "Human review", {
            "text": "The AI system must not decide without human review.",
            "citation_id": "new.md#review@v2", "source": "new.md", "version": "v2",
            "concepts": ["human-review"], "modality": "prohibition", "is_current": True, "authority_rank": 2,
        }))
        graph.add_edge(Edge("clause:old", "doc:old", "DERIVED_FROM"))
        graph.add_edge(Edge("clause:new", "doc:new", "DERIVED_FROM"))
        graph.add_edge(Edge("doc:new", "doc:old", "OVERRIDES"))
        graph.add_edge(Edge("clause:old", "clause:new", "CONTRADICTS"))
        claims = [AtomicClaim(
            id="claim:decision",
            text="The AI system is allowed to decide without human review.",
            kind="decision", decisive=True,
        )]
        cert = build_temporal_drift_certificate(
            "Can the AI decide without human review?", claims, graph, "allowed", ["clause:old", "clause:new"]
        )
        self.assertFalse(cert.stable_across_versions)
        self.assertTrue(cert.retroactive_review_required)
        self.assertEqual(cert.earliest_decision, "allowed")
        self.assertEqual(cert.latest_decision, "needs_review")


if __name__ == "__main__":
    unittest.main()
