# Architecture

VeriWeave-VITA-PRO separates language generation from policy verification. The system treats an LLM answer as a candidate decision that must be independently checked against a typed, version-aware policy graph.

## Processing pipeline

```text
Question + policy corpus
          |
          v
Policy loader and typed property graph
          |
          +-------------------------------+
          |                               |
          v                               v
Initial retrieval                  Independent audit retrieval
          |                               |
          v                               |
Candidate LLM response                    |
          |                               |
          v                               |
Atomic claim extraction                   |
          |                               |
          +---------------+---------------+
                          v
                Claim-level validation
        support / contradiction / applicability
          provenance / version / precedence
                          |
          +---------------+----------------+
          |                                |
          v                                v
Evidence Horizon search          PRO candidate selection
(single omitted clauses)         (diverse, current, authoritative,
                                  independently supported evidence)
          |                                |
          +---------------+----------------+
                          v
              Bounded VITA closure
       singleton and coalitional counterevidence
                          |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
Decision-space       Argumentation      Temporal replay
certificate          certificate        and drift check
       |                  |                  |
       +------------------+------------------+
                          v
              Verification envelope
                          |
                          v
        Trace, metrics, manifest, and report
```

## Policy graph

The graph models:

- documents and policy clauses;
- concepts and applicability relationships;
- policy versions and currency;
- authorities and authority rank;
- support, contradiction, override, supersession, and potential-conflict relations.

The graph is built from `data/policies/` and exported at runtime to `data/graph/evidence_graph.json`. Generated exports are not version-controlled; `data/graph/schema.json` is the durable schema contract.

## Provenance-Robust Optimization

PRO is the recommended evidence-selection strategy. It is deterministic under fixed inputs and scores candidates using a coverage-and-diversity objective that rewards:

- claim and question coverage;
- complete source, version, citation, and graph-path provenance;
- current and higher-authority clauses;
- independent support from different sources;
- contradiction, override, supersession, and modality-risk relations;
- concept diversity;
- low redundancy.

The resulting certificate records selected clauses, marginal gains, provenance completeness, source diversity, independent-support coverage, risk-relation coverage, residual risk mass, and a graph digest.

PRO does not claim globally complete evidence discovery. It makes the bounded selection process explicit, reproducible, and inspectable.

## Evidence Horizon

Evidence Horizon searches outside the initially visible evidence set for omitted clauses that independently change the decision or increase review requirements. It also estimates fragility through bounded evidence-cut analysis.

## VITA closure

VITA performs bounded singleton and coalition tests over omitted evidence. It records permissive and restrictive outcomes, synergistic blind spots, closure convergence, and the effective evidence set.

The closure stops when no candidates remain, no new decision-changing evidence is found, no selected evidence can be added, or the configured round budget is exhausted.

## Argumentation and precedence

Explicit policy relations, version currency, authority rank, scope, and modality are used to resolve competing clauses. The argumentation certificate reports attacks, successful defeats, unresolved conflicts, and conflict freedom.

## Temporal replay

Temporal replay evaluates whether a decision remains valid across policy snapshots. It identifies decision drift, obsolete evidence, and cases that may require retrospective review.

## BPA ablation

Boltzmann Policy Attention remains available as an ablation. It evaluates bounded policy-clause coalitions using unary energies, pairwise policy couplings, and a temperature schedule. BPA results must be reported separately from PRO results.

## System boundary

VeriWeave is research software. It does not replace legal review, policy ownership, access control, model governance, or production authorization. A deployment must add identity, authorization, durable storage, audit protection, operational monitoring, and organization-specific policy lifecycle controls.
