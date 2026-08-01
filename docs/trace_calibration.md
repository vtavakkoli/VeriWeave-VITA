# Trace-calibrated VITA validation

## Purpose

Version 4.2.1 corrects failure modes discovered in the first 300-task model-backed run. Those tasks are a **calibration set**, not an unbiased final test set. Claims about improvement should therefore be based on the untouched tasks `T301`–`T600` and, ideally, additional models and seeds.

## Diagnosed failure modes

The calibration traces showed four systematic problems:

1. Standard VITA treated moderate residual ranked-risk mass as a blocking condition on every task, producing `needs_review` for all 300 cases.
2. VITA-PRO could reward source diversity strongly enough to select stale or weakly related clauses ahead of decisive current-version evidence.
3. Inline citation identifiers were included in lexical comparisons and could distort support, contradiction, and modality analysis.
4. Similar clauses from unrelated sources with different version strings could be treated as unresolved policy alternatives.

Version 4.2.1 separates blocking risks from audit advisories, calibrates normative decisions, tightens precedence, normalizes citations, and adds trace-derived regression tests.

## Held-out run

Run only VITA and VITA-PRO on the untouched second half of the 600-task benchmark:

```bash
veriweave \
  --model gemma4:31b-cloud \
  --task-offset 300 \
  --max-tasks 300 \
  --methods "VeriWeave-VITA,VeriWeave-VITA-PRO"
```

Equivalent environment variables:

```bash
TASK_OFFSET=300 \
MAX_TASKS=300 \
OLLAMA_MODEL=gemma4:31b-cloud \
METHODS="VeriWeave-VITA,VeriWeave-VITA-PRO" \
veriweave
```

The generated manifest records `benchmark_total_tasks`, `task_offset`, `task_end_exclusive`, and `tasks_evaluated`, so the evaluated range is auditable.

## Recommended confirmation matrix

For a publication-quality result, evaluate:

- tasks `T301`–`T600` as the primary held-out set;
- at least three random seeds or repeated cloud-model runs;
- VITA and VITA-PRO with the same model, generation parameters, retrieval budget, and audit;
- the original baselines under the same evaluator when compute permits;
- an additional policy corpus not used to design the rules.

Report decision accuracy, review-routing accuracy, unsupported-claim rate, citation precision and coverage, graph-reference recall, evidence-cut robustness, temporal stability, latency, and the composite score. Also report the distribution of final decisions; a degenerate distribution such as all `needs_review` is a calibration warning even when aggregate accuracy appears competitive.

## Acceptance criteria

The patch should be considered successful only when the held-out run demonstrates all of the following:

- VITA no longer collapses to one decision class;
- review-routing improves without reducing contradiction safety;
- VITA-PRO retains or improves current-version evidence coverage;
- citation coverage improves without lowering citation precision;
- PRO does not lose overall score to VITA by more than the paired confidence interval;
- no fallback model calls occur and `valid_for_paper` is `true`.

A perfect score is not expected or claimed. Remaining errors should be added as new regression fixtures only after the held-out evaluation is frozen and reported.
