# VeriWeave Govern architecture

## Product boundary

VeriWeave Govern is a decision-support and enforcement control plane. It does not replace legal counsel, organizational accountability, identity management, model safety evaluation, or production authorization controls. It creates a deterministic governance gate around AI-agent actions and preserves the evidence used for each decision.

## Request flow

```text
Agent / workflow / API gateway
             |
             v
      Evaluation API
             |
     +-------+--------+
     |                |
Policy matcher   Evidence verifier
     |                |
     +-------+--------+
             |
      Decision reducer
      deny > review > allow
             |
     Human review routing
             |
   Hash-chained audit ledger
```

## Design principles

1. **Fail safe:** an unmatched action is routed to review, not silently allowed.
2. **Deterministic enforcement:** policy matching does not depend on an LLM.
3. **Evidence before execution:** allow rules can be escalated when required evidence is missing or weak.
4. **Versioned accountability:** every response includes the active policy-set hash and policy versions.
5. **Tamper evidence:** audit records form a SHA-256 chain and can optionally carry HMAC signatures.
6. **Human ownership:** review queues are policy-defined and explicit.
7. **Local-first deployment:** the MVP runs without external model or cloud dependencies.

## Integration path

- **Policy-as-Skill:** import governed skills, required evidence, allowed actions and review triggers as policy rules.
- **VeriWeave-VITA-PRO:** replace the deterministic evidence adapter with provenance-robust evidence selection and certificates.
- **Identity:** receive tenant, user, agent and role claims from Keycloak, Entra ID or a gateway such as Kong.
- **Workflow systems:** map review queues to ServiceNow, Jira, BMC Helix or a customer approval service.
- **Policy repositories:** synchronize approved versions from SharePoint, Confluence, Git or document management systems.
- **Audit backends:** export signed records to immutable object storage, SIEM, OpenTelemetry or a tamper-evident ledger.

## Production hardening roadmap

- OIDC authentication, tenant isolation and role-based administration
- PostgreSQL persistence and migrations
- policy approval workflow with four-eyes control
- signed policy bundles and trusted publisher keys
- secret-manager integration and key rotation
- rate limits, idempotency, quotas and workload identity
- OpenTelemetry traces, metrics and structured logs
- encrypted fields and retention policies
- external audit anchoring
- independent threat model, penetration test and supply-chain review
