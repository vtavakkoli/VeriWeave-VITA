from __future__ import annotations

import re
from collections import defaultdict

from app.models import EvidenceAssessment, EvidenceItem


class EvidenceVerifier:
    """Deterministic evidence gate designed to be explainable and testable.

    The existing VeriWeave research methods can later replace or augment this
    adapter, while the governance API and decision contract remain stable.
    """

    def assess(self, evidence: list[EvidenceItem]) -> list[EvidenceAssessment]:
        duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
        for item in evidence:
            duplicate_counts[(item.source_id, item.content.strip().lower())] += 1

        results: list[EvidenceAssessment] = []
        for item in evidence:
            score = 0.0
            reasons: list[str] = []

            score += item.authority / 100.0 * 0.35
            reasons.append(f"authority={item.authority}/100")

            if item.current:
                score += 0.20
                reasons.append("current source version")
            else:
                reasons.append("source marked non-current")

            if item.signed:
                score += 0.15
                reasons.append("signed evidence")
            else:
                reasons.append("unsigned evidence")

            if len(item.content.split()) >= 8:
                score += 0.15
                reasons.append("substantive content")
            else:
                reasons.append("content too short")

            if re.search(r"\b(section|article|control|policy|requirement|clause)\b", item.content, re.I):
                score += 0.10
                reasons.append("contains policy reference language")

            if duplicate_counts[(item.source_id, item.content.strip().lower())] == 1:
                score += 0.05
            else:
                reasons.append("duplicate evidence")

            score = min(round(score, 4), 1.0)
            results.append(
                EvidenceAssessment(
                    evidence_id=item.evidence_id,
                    accepted=score >= 0.55 and item.current,
                    score=score,
                    reasons=reasons,
                )
            )
        return results
