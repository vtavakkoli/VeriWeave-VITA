from pathlib import Path

from app.audit import AuditLedger
from app.engine import GovernanceEngine
from app.evidence import EvidenceVerifier
from app.models import Decision, EvidenceItem, EvaluationRequest
from app.policy import PolicyStore


POLICY_DIR = Path(__file__).resolve().parents[1] / "config" / "policies"


def build_engine(tmp_path: Path) -> GovernanceEngine:
    return GovernanceEngine(
        PolicyStore(POLICY_DIR),
        EvidenceVerifier(),
        AuditLedger(tmp_path / "audit.jsonl", "test-key"),
    )


def strong_evidence(evidence_id: str, evidence_type: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_id="policy-library",
        source_version="2026.1",
        evidence_type=evidence_type,
        content="Policy section 4 requires documented approval and accountable human oversight.",
        authority=90,
        current=True,
        signed=True,
    )


def test_low_risk_read_is_allowed_with_policy_evidence(tmp_path: Path) -> None:
    response = build_engine(tmp_path).evaluate(
        EvaluationRequest(
            agent_id="knowledge-assistant",
            action="summarize",
            context={"impact": "low", "environment": "test"},
            evidence=[strong_evidence("e1", "policy_reference")],
        )
    )
    assert response.decision == Decision.ALLOW
    assert response.audit.signature


def test_missing_evidence_escalates_allow_to_review(tmp_path: Path) -> None:
    response = build_engine(tmp_path).evaluate(
        EvaluationRequest(
            agent_id="knowledge-assistant",
            action="summarize",
            context={"impact": "low"},
        )
    )
    assert response.decision == Decision.REVIEW
    assert "policy_reference" in response.matched_rules[0].missing_evidence


def test_protected_data_transfer_is_denied(tmp_path: Path) -> None:
    response = build_engine(tmp_path).evaluate(
        EvaluationRequest(
            agent_id="export-agent",
            action="send_external",
            resource="customer-records",
            context={"data_classification": "secret", "impact": "critical"},
        )
    )
    assert response.decision == Decision.DENY
    assert response.review_queue == "security-incident"


def test_audit_chain_detects_no_tampering(tmp_path: Path) -> None:
    engine = build_engine(tmp_path)
    request = EvaluationRequest(agent_id="agent", action="unknown", context={})
    engine.evaluate(request)
    engine.evaluate(request)
    assert engine.audit_ledger.verify()["valid"] is True
    assert engine.audit_ledger.verify()["records"] == 2
