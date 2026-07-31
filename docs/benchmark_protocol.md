# Benchmark Protocol

This document defines the minimum protocol for reproducible VeriWeave experiments.

## Benchmark scope

The main benchmark contains 600 held-out evaluation tasks balanced across four task families. These tasks are evaluation cases, not training examples. The repository also includes deterministic mechanism challenges for targeted software validation.

Before a reported run, freeze:

- the benchmark and policy corpus;
- graph construction and relation rules;
- model identifier and generation parameters;
- prompts and structured-output schema;
- retrieval and evidence budgets;
- PRO weights and BPA parameters;
- evaluator thresholds;
- random seeds.

## Compared methods

- Direct LLM
- Text RAG
- Community GraphRAG
- PPR GraphRAG
- Steiner GraphRAG
- VeriWeave-Core
- VeriWeave-Horizon
- VeriWeave-VITA
- VeriWeave-VITA-BPA
- VeriWeave-VITA-PRO

Graph retrieval baselines are controlled approximations and must not be described as official reproductions.

## Equal-audit principle

Every method output receives the same independent post-hoc audit, including claim validation, precedence resolution, Evidence Horizon analysis, bounded VITA testing, argumentation checks, temporal replay, and the common evidence-selection stress test. Method-native certificates are reported separately and do not automatically receive correctness credit.

## Paper-valid execution

Scientific model-backed runs must use:

```text
OLLAMA_ENABLED=true
STRICT_MODEL_RUN=true
```

A run is paper-valid only when `result/manifest.json` contains `"valid_for_paper": true`. This requires:

- all expected model calls to succeed;
- no fallback model outputs;
- no captured execution failures;
- a retained manifest describing the model, seed, budgets, methods, and generation settings.

Offline mode is only a software smoke test. Do not mix offline outputs with model-backed results.

## Primary hypotheses

1. PRO improves decisive-clause and provenance coverage at an equal evidence budget.
2. Independent-source support and diversity penalties reduce redundant evidence selection.
3. Risk-relation coverage improves detection of contradictions, overrides, supersession, and modality conflicts.
4. Evidence Horizon improves detection of individually decision-changing omitted clauses.
5. VITA improves detection of synergistic counterevidence that cannot be exposed by singleton search.
6. Temporal replay improves policy-version and decision-drift detection.
7. PRO should provide stronger concentration and stability than BPA when BPA attention is diffuse.

## Metrics

Report answer-quality, review-routing, provenance, retrieval, robustness, and efficiency metrics. At minimum include:

- decision accuracy;
- review-routing accuracy;
- claim precision and unsupported-claim rate;
- citation precision and claim-level citation coverage;
- provenance completeness and current-evidence rate;
- decisive-clause and graph-reference recall;
- independent-source support and source diversity;
- risk-relation coverage;
- evidence-cut robustness;
- synergistic-blind-spot recall;
- decision-space width and invariance;
- closure convergence;
- argumentation conflict freedom;
- temporal stability and drift detection;
- latency, candidate count, subset tests, and model calls.

For PRO, additionally report marginal gains, selected evidence, residual risk mass, and selection stability. For BPA, report selected attention mass, residual probability mass, normalized entropy, effective sample size, and risk-tail mass.

## Required ablations

1. VeriWeave-VITA deterministic candidate ranking;
2. full VeriWeave-VITA-PRO;
3. PRO without provenance completeness;
4. PRO without authority and currency;
5. PRO without source-diversity reward;
6. PRO without redundancy penalty;
7. PRO without risk-relation coverage;
8. PRO candidate- and evidence-budget sweeps;
9. VeriWeave-VITA-BPA unary-only;
10. BPA with pairwise policy couplings;
11. fixed versus adaptive BPA temperature;
12. no annealing versus annealing;
13. singleton versus pair and triple policy sets;
14. equal evidence-count budget;
15. equal subset-test budget;
16. equal latency budget;
17. corrupted or shuffled graph relations;
18. multiple models and stochastic seeds.

## Statistical analysis

Use paired task-level comparisons, bootstrap confidence intervals, and paired permutation or sign tests. Report per-family and difficulty breakdowns. Correct for multiple comparisons when testing many methods or ablations. Separate development data used for parameter selection from the final held-out evaluation set.

## Reporting requirements

Retain and publish, where permitted:

- the exact commit identifier;
- `manifest.json`;
- model and generation configuration;
- benchmark and policy-corpus hashes;
- task-level metrics;
- failures and excluded cases;
- seeds and statistical-analysis code;
- compute and latency budgets.

Do not reuse results from an older method version as results for a newer method.

## External validity

Strong claims require at least one independently curated policy corpus with genuine amendments, exceptions, cross-references, authority conflicts, and applicability rules. Human annotators should label decisive clauses, interacting clause sets, precedence outcomes, and justified review routing, with inter-annotator agreement reported.
