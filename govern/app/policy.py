from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.models import Decision, EvaluationRequest


class Predicate(BaseModel):
    field: str = Field(min_length=1)
    operator: str = Field(default="eq")
    value: Any = None


class PolicyRule(BaseModel):
    id: str = Field(min_length=1)
    description: str = ""
    when: list[Predicate] = Field(default_factory=list)
    decision: Decision = Decision.REVIEW
    required_evidence: list[str] = Field(default_factory=list)
    min_evidence_score: float = Field(default=0.55, ge=0.0, le=1.0)
    review_queue: str | None = None
    reason: str = "Policy rule matched"


class PolicyDocument(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: str = "active"
    owner: str = "governance"
    effective_from: date | None = None
    tags: list[str] = Field(default_factory=list)
    rules: list[PolicyRule] = Field(min_length=1)


@dataclass(frozen=True)
class LoadedPolicy:
    document: PolicyDocument
    content_hash: str
    path: Path


class PolicyLoadError(RuntimeError):
    pass


class PolicyStore:
    def __init__(self, policy_dir: Path) -> None:
        self.policy_dir = policy_dir
        self._policies: list[LoadedPolicy] = []
        self.reload()

    @property
    def policies(self) -> list[LoadedPolicy]:
        return list(self._policies)

    @property
    def policy_set_hash(self) -> str:
        payload = "|".join(sorted(policy.content_hash for policy in self._policies))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def reload(self) -> None:
        loaded: list[LoadedPolicy] = []
        if not self.policy_dir.exists():
            raise PolicyLoadError(f"Policy directory does not exist: {self.policy_dir}")

        for path in sorted(self.policy_dir.glob("*.y*ml")):
            raw_text = path.read_text(encoding="utf-8")
            try:
                raw = yaml.safe_load(raw_text)
                document = PolicyDocument.model_validate(raw)
            except (yaml.YAMLError, ValidationError) as exc:
                raise PolicyLoadError(f"Invalid policy {path}: {exc}") from exc

            if document.status.lower() != "active":
                continue
            loaded.append(
                LoadedPolicy(
                    document=document,
                    content_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    path=path,
                )
            )

        if not loaded:
            raise PolicyLoadError(f"No active YAML policies found in {self.policy_dir}")
        self._policies = loaded

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.document.id,
                "name": item.document.name,
                "version": item.document.version,
                "owner": item.document.owner,
                "tags": item.document.tags,
                "rules": len(item.document.rules),
                "content_hash": item.content_hash,
            }
            for item in self._policies
        ]


def request_field(request: EvaluationRequest, dotted_field: str) -> Any:
    roots: dict[str, Any] = {
        "tenant_id": request.tenant_id,
        "agent_id": request.agent_id,
        "action": request.action,
        "resource": request.resource,
        "purpose": request.purpose,
        "context": request.context,
    }
    parts = dotted_field.split(".")
    current: Any = roots.get(parts[0])
    for part in parts[1:]:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def predicate_matches(request: EvaluationRequest, predicate: Predicate) -> bool:
    actual = request_field(request, predicate.field)
    expected = predicate.value
    operator = predicate.operator.lower()

    if operator == "exists":
        return actual is not None
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator == "in":
        return actual in expected if isinstance(expected, (list, tuple, set)) else False
    if operator == "not_in":
        return actual not in expected if isinstance(expected, (list, tuple, set)) else False
    if operator == "contains":
        return expected in actual if isinstance(actual, (str, list, tuple, set, dict)) else False
    if operator == "starts_with":
        return isinstance(actual, str) and isinstance(expected, str) and actual.startswith(expected)
    if operator == "gte":
        return actual is not None and actual >= expected
    if operator == "lte":
        return actual is not None and actual <= expected
    if operator == "truthy":
        return bool(actual) is bool(expected)
    raise PolicyLoadError(f"Unsupported predicate operator: {predicate.operator}")


def rule_matches(request: EvaluationRequest, rule: PolicyRule) -> bool:
    return all(predicate_matches(request, predicate) for predicate in rule.when)


def canonical_policy_json(policy: PolicyDocument) -> str:
    return json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
