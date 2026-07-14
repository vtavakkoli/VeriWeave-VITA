# Benchmark Protocol

## Main benchmark

The repository retains 600 held-out evaluation tasks balanced across four task families. They must not be described as training examples. Freeze graph construction, prompts, energy weights, temperatures, budgets, and evaluator thresholds before the reported run.

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

## Equal-audit principle

All outputs receive the same independent verifier, horizon test, bounded BPA audit, VITA test, argumentation check, and temporal replay. Method-native certificates are reported separately and do not automatically earn correctness credit.

## Primary hypotheses

- BPA improves decisive policy-clause recall at equal evidence budgets.
- Pairwise couplings improve synergistic blind-spot recall over unary-only attention.
- Adaptive annealing lowers residual VITA risk at equal validation-call budgets.
- BPA helps most on version, exception, scope, and conflict tasks, with limited gains on simple lexical tasks.

## Metrics

Report answer and verification metrics plus:

- selected attention mass;
- residual probability mass;
- normalized attention entropy;
- effective sample size;
- risk-tail mass;
- decisive-clause recall within the selected pool;
- synergistic-blind-spot recall;
- decision-space width and invariance;
- closure convergence;
- argumentation conflict-freedom;
- temporal stability;
- latency, enumerated coalitions, and validation calls.

For synthetic hidden-clause tasks, also report candidate rank, marginal attention rank, top-coalition recall, and calibration of residual mass against missed decisive clauses.

## Required ablations

1. deterministic top-*k* VITA ranking;
2. independent softmax over unary clause utilities;
3. BPA unary-only;
4. BPA with policy couplings;
5. fixed temperature;
6. adaptive temperature without annealing;
7. full adaptive annealing;
8. no contradiction/override couplings;
9. no redundancy repulsion;
10. singleton vs pair vs triple policy sets;
11. equal evidence-count budget;
12. equal subset-test budget;
13. equal latency budget;
14. corrupted graph edges;
15. temperature and attention-mass sweeps.

## Statistical analysis

Use paired task-level comparisons, bootstrap confidence intervals, paired sign or permutation tests, per-family and difficulty breakdowns, and multiple seeds where generation is stochastic. For BPA hyperparameters, separate development and held-out test sets; do not tune on the final 600-task results.

## External validity

Add at least one independently curated policy corpus with genuine amendments, exceptions, cross-references, and authority conflicts. Human annotators should label decisive clauses, interacting clause sets, precedence outcomes, and justified review routing. Report inter-annotator agreement.
