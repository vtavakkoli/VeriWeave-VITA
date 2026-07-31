import unittest

from veriweave.graph import Edge, Node, PropertyGraph
from veriweave.models import AtomicClaim
from veriweave.provenance_robust_selection import select_provenance_robust_candidates
from veriweave.retrieval import Evidence


class ProvenanceRobustSelectionTests(unittest.TestCase):
    def test_prefers_current_independent_sources_over_duplicate_old_clause(self):
        graph = PropertyGraph()
        for clause_id, text, source, version, current, authority in [
            ("c:seed", "AI decisions require review.", "seed.md", "v1", True, 2),
            ("c:dup", "AI decisions require review.", "seed.md", "v1", True, 2),
            ("c:independent", "A human reviewer must approve AI decisions.", "regulator.md", "v2", True, 3),
            ("c:old", "AI decisions may proceed automatically.", "legacy.md", "v1", False, 1),
        ]:
            graph.add_node(Node(clause_id, "Clause", clause_id, {
                "text": text,
                "citation_id": f"{source}#{clause_id}@{version}",
                "source": source,
                "version": version,
                "concepts": ["human-review"],
                "modality": "obligation" if "require" in text or "must" in text else "permission",
                "is_current": current,
                "authority_rank": authority,
            }))
        graph.add_edge(Edge("c:old", "c:independent", "CONTRADICTS"))
        seed = Evidence("c:seed", "seed.md#c:seed@v1", "AI decisions require review.", 0.9,
                        "seed.md", "v1", ["human-review"], ["c:seed"], "retrieved", "test", "obligation", True, 2)
        candidates = [
            Evidence("c:dup", "seed.md#c:dup@v1", "AI decisions require review.", 0.95,
                     "seed.md", "v1", ["human-review"], ["c:dup"], "candidate", "test", "obligation", True, 2),
            Evidence("c:independent", "regulator.md#c:independent@v2", "A human reviewer must approve AI decisions.", 0.82,
                     "regulator.md", "v2", ["human-review"], ["c:independent"], "candidate", "test", "obligation", True, 3),
            Evidence("c:old", "legacy.md#c:old@v1", "AI decisions may proceed automatically.", 0.70,
                     "legacy.md", "v1", ["human-review"], ["c:old"], "candidate", "test", "permission", False, 1),
        ]
        claims = [AtomicClaim("claim:1", "AI decisions require human review.", decisive=True)]
        selected, certificate = select_provenance_robust_candidates(
            "May an AI decide without human review?", claims, [seed], candidates, graph, selection_budget=2
        )
        ids = [item.clause_id for item in selected]
        self.assertIn("c:independent", ids)
        self.assertGreater(certificate.provenance_completeness, 0.9)
        self.assertGreater(certificate.source_diversity, 0.0)


if __name__ == "__main__":
    unittest.main()
