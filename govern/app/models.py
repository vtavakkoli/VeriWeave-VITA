from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Decision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    DENY = "deny"


class EvidenceItem(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=200)
    source_version: str = Field(min_length=1, max_length=100)
    evidence_type: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=20_000)
    authority: int = Field(default=50, ge=0, le=100)
    current: bool = True
    signed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=100)
    agent_id: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=200)
    resource: str = Field(default="", max_length=500)
    purpose: str = Field(default="", max_length=1_000)
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=100)
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=200)

    @field_validator("action", "resource", "purpose")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class EvidenceAssessment(BaseModel):
    evidence_id: str
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class RuleResult(BaseModel):
    policy_id: str
    policy_version: str
    rule_id: str
    matched: bool
    decision: Decision
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    review_queue: str | None = None


class AuditEnvelope(BaseModel):
    record_id: str
    created_at: datetime
    record_hash: str
    previous_hash: str
    signature: str | None = None


class EvaluationResponse(BaseModel):
    request_id: str
    decision: Decision
    reasons: list[str]
    matched_rules: list[RuleResult]
    evidence_assessments: list[EvidenceAssessment]
    policy_set_hash: str
    audit: AuditEnvelope
    review_queue: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
