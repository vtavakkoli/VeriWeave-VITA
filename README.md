# VeriWeave-VITA-BPA

**VITA** stands for **Verification-Impact, Temporal, and Argumentation** reasoning.

**VeriWeave-VITA-BPA** is a research prototype for auditing LLM decisions over evolving policy graphs when decisive evidence may be outside the initially retrieved subgraph, may arise only from a combination of clauses, and cannot be covered reliably by deterministic top-*k* expansion.

> Paper direction: **VeriWeave-VITA-BPA: Verification-Impact, Temporal and Argumentation Reasoning with Boltzmann Policy Attention for Coalitional Counterevidence Certification over Evolving Policy Graphs**

The central question is:

> **Which omitted policy-clause set deserves the limited verification budget, and how stable is the decision under interacting counterevidence and policy evolution?**

## Main addition: Boltzmann Policy Attention

Boltzmann Policy Attention (BPA) is an inference-time energy model over **sets of policy clauses**, not a modification of Transformer token attention. For a candidate coalition \(S\), the implementation defines:

```text
P(S | q, d, G) ∝ exp(U(S) / τ)
```

where:

- `q` is the question and extracted claims;
- `d` is the provisional decision;
- `G` is the policy property graph;
- `U(S)` combines unary clause utility and pairwise policy couplings;
- `τ` is an adaptive, annealed temperature.

Unary utility considers semantic relevance, decision-conditioned risk, authority, temporal currency, structural graph proximity, and concept novelty. Pairwise couplings reward clause sets that should be inspected together, including:

- explicit contradiction;
- `OVERRIDES` or version succession;
- opposed normative modalities;
- shared policy concepts;
- same-source version pairs;
- temporal contrast.

Near-duplicate clauses receive a repulsive coupling. The algorithm exactly enumerates bounded coalitions, calculates marginal clause attention, and selects a clause set that captures a configurable share of normalized marginal clause-attention mass within a hard evidence budget; it separately reports the probability of coalitions fully covered by that set.

### Why this is different from softmax top-*k*

A top-*k* retriever scores clauses independently. BPA assigns probability to **interacting clause coalitions**. A moderate-scoring clause can therefore receive high marginal attention because it completes a high-risk version pair, contradiction, or scope–exception interaction.

### Adaptive annealing

A flatter unary landscape or denser interaction graph increases the exploration temperature. Each closure round then cools the temperature:

```text
τ_r = max(τ_min, τ_adaptive × anneal_rate^(r-1))
```

Early rounds explore competing policy sets; later rounds concentrate on the most consequential VITA candidates.

### BPA certificate

Every BPA run can emit a machine-readable certificate containing:

- temperature schedule;
- unary energies and their components;
- pairwise couplings and reasons;
- highest-probability coalitions;
- marginal clause attention;
- selected marginal attention mass;
- covered-coalition probability;
- residual probability mass;
- normalized entropy;
- effective sample size;
- risk-tail mass;
- free energy and graph digest.

See [`docs/boltzmann_policy_attention.md`](docs/boltzmann_policy_attention.md).

## Full verification architecture

BPA extends, rather than replaces, the previous mechanisms:

1. **Claim Verification Envelope** — decomposes the answer and validates support, contradiction, applicability, temporal validity, and provenance.
2. **Evidence Horizon Search** — searches outside the initial subgraph for individually decision-changing clauses.
3. **Coalitional Counterevidence Closure** — tests interacting omitted clauses and detects synergistic blind spots.
4. **Two-Sided VITA Decision-Space** — reports the most permissive and most restrictive decisions reachable under tested evidence subsets.
5. **Precedence-Grounded Argumentation** — resolves attacks using explicit overrides, version currency, authority, applicability, and modality.
6. **Temporal Decision-Drift Replay** — checks whether an earlier answer remains valid across policy snapshots.
7. **Boltzmann Policy Attention** — allocates the bounded VITA search budget over interacting policy-clause sets and quantifies unselected probability mass.

## Controlled methods

The benchmark now supports nine methods:

1. Direct LLM
2. Text RAG
3. Community GraphRAG
4. PPR GraphRAG
5. Steiner GraphRAG
6. VeriWeave-Core
7. VeriWeave-Horizon
8. VeriWeave-VITA
9. **VeriWeave-VITA-BPA**

The graph baselines are transparent standard-library approximations for controlled comparison; they are not official reproductions.

The principal ablation is:

```text
Core
  + singleton omitted-evidence testing             = Horizon
  + coalitional closure, argumentation, drift      = VITA
  + energy-based coalition attention and annealing = VITA-BPA
```

## Fair evaluation

Every method receives the same independent post-hoc audit, including the same bounded BPA stress test. Method-produced certificates are reported separately. The evaluator does not add citations, does not replace the model decision with keyword rules, and separates answer quality from verification, provenance, retrieval, and robustness metrics.

Required BPA ablations include:

- deterministic VITA ranking versus BPA;
- unary-only BPA versus pairwise couplings;
- fixed versus adaptive temperature;
- no annealing versus annealing;
- singleton versus pair and triple policy sets;
- equal candidate, subset-test, and latency budgets;
- shuffled contradiction/version edges;
- temperature and attention-mass sensitivity.

## Installation and execution

```bash
python -m pip install -e .
```

Run the full benchmark:

```bash
PYTHONPATH=src python -m veriweave.main
```

Example with Ollama:

```bash
OLLAMA_MODEL=gemma4:31b PYTHONPATH=src python -m veriweave.main
```

Run only the new method:

```bash
METHODS=VeriWeave-VITA-BPA \
BOLTZMANN_TEMPERATURE=0.75 \
BOLTZMANN_ATTENTION_MASS=0.90 \
PYTHONPATH=src python -m veriweave.main
```

Offline software smoke test:

```bash
OLLAMA_ENABLED=false MAX_TASKS=8 PYTHONPATH=src python -m veriweave.main
```

CLI example:

```bash
veriweave --model gemma4:31b --temperature 0.75 --attention-mass 0.90
```

## Deterministic challenges

```bash
PYTHONPATH=src python -m veriweave.horizon_challenge
PYTHONPATH=src python -m veriweave.vita_challenge
PYTHONPATH=src python -m veriweave.boltzmann_challenge
```

The BPA challenge currently passes **4/4** deterministic mechanism checks:

- the policy-set distribution is normalized;
- lower temperature produces more concentrated attention;
- pairwise coupling surfaces a current counterrule and its obsolete predecessor;
- VITA integration emits a BPA certificate and exposes restrictive decisions.

These are software and mechanism checks, not LLM-quality results.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The repository contains **20 passing tests**.

## New metrics

Common-audit metrics:

- `attention_selected_mass`
- `attention_residual_mass`
- `attention_entropy`
- `attention_effective_sample_size`
- `attention_risk_tail_mass`

Method-specific metrics:

- `method_attention_selected_mass`
- `method_attention_residual_mass`
- `method_attention_entropy`
- `method_attention_effective_sample_size`
- `method_attention_risk_tail_mass`
- `method_attention_selected_clauses`

These complement decision-space width, closure convergence, residual risk, synergistic blind spots, argumentation conflict-freedom, temporal stability, and evidence-cut robustness.

## Repository structure

```text
src/veriweave/
  boltzmann_policy_attention.py  # exact policy-set energy model and certificate
  vita.py                    # coalitional closure and two-sided VITA decision space
  argumentation.py               # precedence-grounded evidence arguments
  temporal.py                    # version replay and decision drift
  horizon.py                     # singleton omitted-evidence analysis
  envelope.py                    # multi-certificate verification envelope
  audit.py                       # identical post-hoc audit for all methods
  evaluators.py                  # separated quality and robustness metrics
  boltzmann_challenge.py         # deterministic BPA challenge
```

## Research-use warning

The included corpus is synthetic or real-world-inspired and de-identified. Exact coalition enumeration is bounded and does not prove global completeness. The BPA energy terms are interpretable hand-designed features, not learned optimal weights. Strong empirical claims require a held-out independently curated policy corpus, human annotations of decisive clause sets, multiple models and seeds, faithful external baselines, equal-compute controls, and preregistered hyperparameters. Earlier ClaimGraph, VeriWeave-Horizon, or VeriWeave-VITA scores must not be reused as VITA-BPA results.
