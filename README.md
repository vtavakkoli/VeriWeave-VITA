# VeriWeave-VITA-PRO

**VITA** means **Verification-Impact, Temporal, and Argumentation** reasoning.

**VeriWeave-VITA-PRO** replaces diffuse Boltzmann coalition attention with
**Provenance-Robust Optimization (PRO)**: a deterministic, diversity-aware
selector that prioritizes current, authoritative, independently sourced policy
evidence while still surfacing contradictions, overrides, and version risks.

> Proposed paper title: **VeriWeave-VITA-PRO: Provenance-Robust Policy Evidence Optimization and Counterfactual Certification for LLM Decisions over Evolving Graphs**

## Why PRO replaces BPA

The completed 600-task run showed that VITA-BPA improved citation coverage and
raw evidence-cut robustness, but reduced decision accuracy and human-review
routing compared with VITA. The attention distribution was also almost uniform,
so it did not reliably focus the verification budget on a small decisive set.

PRO uses a deterministic submodular-style objective instead of a Boltzmann
probability distribution. It rewards:

- claim and question coverage;
- complete source, version, citation, and graph-path provenance;
- current and higher-authority clauses;
- independent support from different sources;
- contradiction, override, supersession, and modality-risk relations;
- concept diversity;
- low redundancy between selected clauses.

The selector emits a machine-readable certificate containing the selected
clauses, marginal scores, provenance completeness, current-evidence rate,
source diversity, independent-support score, risk-relation coverage, and
residual risk mass.

## Important evaluation corrections

Version 4.2.0 also fixes three methodological problems:

1. **Strict model execution.** A transient Ollama failure is retried and does
   not disable the model for the rest of the run. In strict mode, exhausted
   retries stop the experiment instead of silently mixing LLM and fallback
   outputs.
2. **Claim-level citation coverage.** A supported claim is covered when at
   least one valid winning citation is supplied; the model is no longer
   penalized for not citing every equivalent passage.
3. **Assessable evidence cuts.** Evidence-cut robustness is zero when no
   independently supported claim exists. Cuts remove independent source groups,
   not merely individual clauses from the same source.

PRO also uses the same candidate-generation prompt as the other VeriWeave
variants, avoiding prompt wording as a confound in the ablation.

## Methods

The benchmark supports:

1. Direct LLM
2. Text RAG
3. Community GraphRAG
4. PPR GraphRAG
5. Steiner GraphRAG
6. VeriWeave-Core
7. VeriWeave-Horizon
8. VeriWeave-VITA
9. VeriWeave-VITA-BPA
10. **VeriWeave-VITA-PRO**

## Run the recommended comparison

```bash
python -m pip install -e .

OLLAMA_MODEL=qwen3:30b-a3b \
QWEN_NO_THINK=true \
OLLAMA_TEMPERATURE=0.7 \
OLLAMA_TOP_P=0.8 \
OLLAMA_TOP_K=20 \
STRICT_MODEL_RUN=true \
METHODS="VeriWeave-VITA,VeriWeave-VITA-BPA,VeriWeave-VITA-PRO" \
PYTHONPATH=src \
python -m veriweave.main
```

Use your installed Ollama model identifier if it differs. The client adds
`/no_think` automatically for Qwen3 when `QWEN_NO_THINK=true`, and the
default sampling parameters follow the official non-thinking profile. They can
be overridden through environment variables for model ablations.

## Full benchmark

```bash
OLLAMA_ENABLED=true \
STRICT_MODEL_RUN=true \
OLLAMA_MAX_RETRIES=3 \
OLLAMA_RETRY_BACKOFF_SECONDS=1.0 \
PYTHONPATH=src \
python -m veriweave.main
```

A paper-valid run must finish with:

```json
"valid_for_paper": true
```

in `result/manifest.json`, with zero fallback calls.

## Offline software smoke test

```bash
OLLAMA_ENABLED=false \
STRICT_MODEL_RUN=false \
MAX_TASKS=8 \
METHODS="VeriWeave-VITA,VeriWeave-VITA-BPA,VeriWeave-VITA-PRO" \
PYTHONPATH=src \
python -m veriweave.main
```

Offline outputs verify software only and must not be reported as experimental
LLM results.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Version 4.2.0 contains **24 passing tests**, including regression tests for
zero-support evidence cuts, claim-level citation coverage, and PRO selection.

## Key files

```text
src/veriweave/provenance_robust_selection.py  # PRO selector and certificate
src/veriweave/vita.py                         # VITA closure with PRO or BPA
src/veriweave/horizon.py                      # source-group evidence cuts
src/veriweave/evaluators.py                   # corrected provenance metrics
src/veriweave/ollama_client.py                # retries and strict execution
src/veriweave/agents.py                       # shared generation prompt
```

## Research-use warning

The included policy corpus and benchmark are synthetic or real-world-inspired.
PRO is deterministic but does not prove global completeness. Strong scientific
claims require a fresh complete model-backed run, multiple seeds and models,
human-validated policy references, faithful external baselines, and equal
candidate and latency budgets. Do not reuse the mixed fallback run as final
paper evidence.
