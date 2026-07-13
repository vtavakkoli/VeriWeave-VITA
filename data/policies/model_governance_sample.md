# Model Governance and Auditability Policy
version: v1

## model-governance
AI-assisted decisions must record model identifier, model version, prompt version or graph-schema or validation-rule version, configuration, timestamp, and responsible owner. A decision that cannot be reproduced from the recorded evidence and configuration is not considered governance-ready.

## evidence-grounding
Policy-aware AI systems must ground every recommendation in retrieved policy evidence. Unsupported claims, uncited recommendations, or claims that contradict cited evidence must be flagged for review.

## access-control
Policy repositories may contain restricted documents. Retrieval must respect user role, department scope, and document sensitivity. Sensitive personal data must not be sent to an external service unless there is explicit approval and documented safeguards.

## audit-trail
The audit trail must include question, selected graph query and validation path, evidence citations, reasoning summary, final decision, confidence, human-review flag, policy versions, policy hashes, and timestamp. Audit records must be exportable for internal or external review.
