# Cybersecurity and Access-Control Policy
version: v1
domain: cybersecurity_access_control

## access-control
Policy-aware AI tools must enforce role-based access, least privilege, and source-level authorization. Retrieval must not expose policy documents or case records outside the user's approved scope.

## audit-trail
Every AI-assisted decision record must store timestamp, question, selected graph query and validation path, retrieved evidence, citations, final decision, confidence, human-review flag, prompt version, model version, policy versions, policy hashes, and responsible owner.

## evidence-grounding
Generated decisions must cite supporting policy evidence. Unsupported claims, irrelevant citations, and citation-decision mismatches require correction or human review.
