# VeriWeave Govern

**VeriWeave Govern** is a local-first policy enforcement and evidence governance control plane for enterprise AI agents.

It turns approved policies into deterministic runtime decisions and produces a traceable record of:

- what an agent attempted;
- which policy version matched;
- what evidence was supplied;
- whether the evidence was sufficiently trustworthy;
- why the action was allowed, denied, or routed to human review;
- which review queue owns the decision;
- the hash-chained audit record created for the evaluation.

The product direction combines the governance concepts of **Policy-as-Skill** with the provenance and verification methods developed in **VeriWeave-VITA-PRO**.

> Status: product-quality MVP and integration foundation. It is not a legal compliance certification, autonomous legal decision maker, or production security boundary by itself.

## Why it is different

Most governance tools document AI systems after deployment. VeriWeave Govern is designed to sit **in the execution path** of an AI agent or automated workflow.

```text
Agent requests an action
          |
          v
VeriWeave Govern
  - match active policy rules
  - validate required evidence
  - enforce deny/review/allow
  - route accountable human review
  - append tamper-evident audit record
          |
          v
Action executes only under the organization's controls
```

## Included MVP capabilities

- FastAPI governance service with OpenAPI documentation
- YAML policy bundles with version, owner, status, tags and rules
- safe predicate DSL without Python `eval`
- deterministic decision precedence: `deny > review > allow`
- fail-safe review when no policy matches
- evidence quality scoring and required-evidence gates
- explicit human-review queues
- policy-set and policy-content hashes
- append-only SHA-256 audit chain
- optional HMAC audit signatures
- audit-chain verification endpoint
- browser dashboard for interactive evaluation
- hardened Docker Compose baseline
- unit tests for allow, escalation, denial and audit integrity

## Run with Docker

```bash
cd govern
cp .env.example .env
# Replace the development signing key in .env
docker compose up --build -d
```

Open:

- Dashboard: `http://localhost:8080`
- OpenAPI: `http://localhost:8080/docs`
- Health: `http://localhost:8080/health`

Stop:

```bash
docker compose down
```

## Run locally

```bash
cd govern
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest -q
python -m app.main
```

## Example evaluation

```bash
curl -X POST http://localhost:8080/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "procurement-agent",
    "action": "summarize",
    "context": {
      "impact": "low",
      "environment": "test",
      "data_classification": "internal"
    },
    "evidence": [{
      "evidence_id": "ev-001",
      "source_id": "approved-policy-library",
      "source_version": "2026.1",
      "evidence_type": "policy_reference",
      "content": "Policy section 4 permits read-only summarization with recorded evidence.",
      "authority": 90,
      "current": true,
      "signed": true
    }]
  }'
```

A response contains the final decision, matched policy rules, evidence assessments, policy-set hash, review queue and audit envelope.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and policy health |
| `GET` | `/v1/policies` | Active policy metadata and hashes |
| `POST` | `/v1/policies/reload` | Reload policy bundles atomically |
| `POST` | `/v1/evaluate` | Evaluate an agent action |
| `GET` | `/v1/audit` | Recent audit records and chain integrity |
| `GET` | `/v1/audit/verify` | Verify the complete audit chain |

## Policy example

```yaml
id: enterprise-ai-governance
name: Enterprise AI Governance Baseline
version: 1.0.0
status: active
owner: AI Governance Office
rules:
  - id: high-impact-human-review
    when:
      - field: context.impact
        operator: in
        value: [high, critical]
    decision: review
    required_evidence:
      - business_justification
      - risk_assessment
    min_evidence_score: 0.65
    review_queue: ai-governance-board
    reason: High-impact AI actions require documented evidence and human oversight.
```

Supported operators are `exists`, `eq`, `neq`, `in`, `not_in`, `contains`, `starts_with`, `gte`, `lte`, and `truthy`.

## Commercial product path

The MVP is deliberately aligned with the needs of public-sector organizations, banks, insurers, healthcare providers, utilities and industrial operators:

1. on-premises or sovereign-cloud deployment;
2. approved policy registry and version lifecycle;
3. runtime agent-action enforcement;
4. human approval workflows;
5. evidence and contradiction verification;
6. signed, exportable audit records;
7. integrations with enterprise identity, ticketing and document systems.

The next product milestone should add OIDC, PostgreSQL, multi-tenancy, policy approvals, ServiceNow/Jira/BMC connectors, a VeriWeave PRO evidence adapter, and external audit anchoring.

See [Architecture](docs/ARCHITECTURE.md) for the system boundary, integration model and production-hardening roadmap.
