from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.audit import AuditLedger
from app.evidence import EvidenceVerifier
from app.models import Decision, EvaluationRequest, EvaluationResponse, RuleResult
from app.policy import PolicyStore, rule_matches


_DECISION_PRIORITY = {Decision.ALLOW: 0, Decision.REVIEW: 1, Decision.DENY: 2}


class GovernanceEngine:
    def __init__(
        self,
        policy_store: PolicyStore,
        evidence_verifier: EvidenceVerifier,
        audit_ledger: AuditLedger,
    ) -> None:
        self.policy_store = policy_store
        self.evidence_verifier = evidence_verifier
        self.audit_ledger = audit_ledger

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        assessments = self.evidence_verifier.assess(request.evidence)
        assessment_by_id = {item.evidence_id: item for item in assessments}
        evidence_by_type: dict[str, list[str]] = defaultdict(list)
        for item in request.evidence:
            evidence_by_type[item.evidence_type].append(item.evidence_id)

        rule_results: list[RuleResult] = []
        for loaded in self.policy_store.policies:
            policy = loaded.document
            for rule in policy.rules:
                if not rule_matches(request, rule):
                    continue

                missing: list[str] = []
                weak: list[str] = []
                for evidence_type in rule.required_evidence:
                    evidence_ids = evidence_by_type.get(evidence_type, [])
                    if not evidence_ids:
                        missing.append(evidence_type)
                        continue
                    if not any(
                        assessment_by_id[evidence_id].accepted
                        and assessment_by_id[evidence_id].score >= rule.min_evidence_score
                        for evidence_id in evidence_ids
                    ):
                        weak.append(evidence_type)

                decision = rule.decision
                reasons = [rule.reason]
                if missing or weak:
                    if decision == Decision.ALLOW:
                        decision = Decision.REVIEW
                    reasons.extend(
                        [
                            *(f"Missing required evidence: {item}" for item in missing),
                            *(f"Evidence below trust threshold: {item}" for item in weak),
                        ]
                    )

                rule_results.append(
                    RuleResult(
                        policy_id=policy.id,
                        policy_version=policy.version,
                        rule_id=rule.id,
                        matched=True,
                        decision=decision,
                        reasons=reasons,
                        missing_evidence=[*missing, *weak],
                        review_queue=rule.review_queue,
                    )
                )

        if not rule_results:
            final_decision = Decision.REVIEW
            reasons = ["No active policy rule matched; fail-safe human review required"]
            review_queue = "governance-default"
        else:
            final_decision = max(
                (result.decision for result in rule_results), key=_DECISION_PRIORITY.__getitem__
            )
            reasons = _unique(reason for result in rule_results for reason in result.reasons)
            review_queue = next(
                (
                    result.review_queue
                    for result in rule_results
                    if result.decision in {Decision.REVIEW, Decision.DENY} and result.review_queue
                ),
                None,
            )

        audit_payload = {
            "request": request.model_dump(mode="json"),
            "decision": final_decision,
            "reasons": reasons,
            "matched_rules": [item.model_dump(mode="json") for item in rule_results],
            "evidence_assessments": [item.model_dump(mode="json") for item in assessments],
            "policy_set_hash": self.policy_store.policy_set_hash,
            "review_queue": review_queue,
        }
        audit = self.audit_ledger.append(audit_payload)

        return EvaluationResponse(
            request_id=request.request_id,
            decision=final_decision,
            reasons=reasons,
            matched_rules=rule_results,
            evidence_assessments=assessments,
            policy_set_hash=self.policy_store.policy_set_hash,
            audit=audit,
            review_queue=review_queue,
        )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
