# VeriWeave-VITA-PRO

[![CI](https://github.com/vtavakkoli/VeriWeave-VITA/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/VeriWeave-VITA/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Research software](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#research-scope-and-limitations)

**VeriWeave-VITA-PRO** is a research framework for evaluating and auditing LLM decisions over evolving policy graphs. It combines provenance-aware evidence selection, counterfactual evidence search, bounded coalitional stress testing, precedence-grounded argumentation, and temporal policy replay.

**VITA** stands for **Verification-Impact, Temporal, and Argumentation** reasoning. The recommended method, **PRO** (**Provenance-Robust Optimization**), deterministically selects a compact, diverse, and auditable evidence set while retaining the earlier Boltzmann Policy Attention (BPA) method as a controlled ablation.

> Proposed paper title: **VeriWeave-VITA-PRO: Provenance-Robust Policy Evidence Optimization and Counterfactual Certification for LLM Decisions over Evolving Graphs**

## Why VeriWeave

Conventional RAG pipelines retrieve passages and generate an answer. VeriWeave adds an independent verification path that asks whether decisive evidence was omitted, superseded, contradicted, or supported only by redundant sources.

```text
Question + policy corpus
          |
          v
Typed policy graph + initial retrieval
          |
          v
Candidate answer and atomic claims
          |
          +-----------------------------+
          |                             |
          v                             v
Provenance-robust selection      Evidence-horizon search
          |                             |
          +--------------+--------------+
                         v
          Bounded VITA coalition testing
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
  Precedence &      Argumentation     Temporal replay
  claim checks       certificate      and drift check
        |                |                |
        +----------------+----------------+
                         v
            Auditable verification envelope
```

## Core capabilities

- Typed property graph for documents, clauses, concepts, versions, and authorities.
- Multiple retrieval baselines plus VeriWeave-specific evidence retrieval.
- Claim-level support, contradiction, applicability, and provenance validation.
- Evidence Horizon search for individually decision-changing omitted clauses.
- VITA closure for synergistic counterevidence and bounded evidence coalitions.
- PRO selection that rewards authority, currency, independent support, diversity, and risk-relation coverage.
- Precedence-grounded argumentation and policy-version replay.
- Common post-hoc auditing across every compared method.
- Machine-readable manifests, traces, metrics, certificates, and an HTML report.
- Strict model-backed execution that prevents silent fallback contamination in paper runs.

## Compared methods

| Family | Method | Purpose |
|---|---|---|
| Generation | Direct LLM | Closed-book baseline |
| Retrieval | Text RAG | Lexical policy retrieval |
| Graph retrieval | Community GraphRAG | Concept-community baseline |
| Graph retrieval | PPR GraphRAG | Personalized-PageRank baseline |
| Graph retrieval | Steiner GraphRAG | Connected-subgraph heuristic |
| Verification | VeriWeave-Core | Claim validation and precedence resolution |
| Counterfactual | VeriWeave-Horizon | Singleton omitted-evidence search |
| Coalitional | VeriWeave-VITA | Bounded coalition, argumentation, and temporal analysis |
| Ablation | VeriWeave-VITA-BPA | Boltzmann policy-set attention |
| Recommended | **VeriWeave-VITA-PRO** | Deterministic provenance-robust evidence selection |

The GraphRAG baselines are transparent research approximations for controlled comparison; they are not official reproductions of external systems.

## Quick start

### Requirements

- Python 3.11 or newer
- Ollama only for model-backed experiments
- Docker and Docker Compose are optional

### Local installation

```bash
git clone https://github.com/vtavakkoli/VeriWeave-VITA.git
cd VeriWeave-VITA
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```

### Offline software smoke test

```bash
veriweave \
  --offline \
  --max-tasks 8 \
  --methods "VeriWeave-Core,VeriWeave-VITA-PRO"
```

Offline mode validates the software pipeline only. Its outputs are **not** model-quality results and must not be reported as experimental LLM findings.

### Model-backed comparison

Start Ollama with an installed model, then run:

```bash
veriweave \
  --model qwen3:30b-a3b \
  --methods "VeriWeave-VITA,VeriWeave-VITA-BPA,VeriWeave-VITA-PRO"
```

The model identifier is configurable. See [`.env.example`](.env.example) for generation parameters, retry behavior, benchmark paths, evidence budgets, and ablation settings.

A model-backed run is paper-valid only when `result/manifest.json` reports:

```json
{
  "valid_for_paper": true
}
```

This additionally requires zero recorded failures and zero fallback model calls.

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Ollama is expected on the host through `host.docker.internal` by default. Generated files are written to `result/`.

## Tests and deterministic challenges

```bash
python -m unittest discover -s tests -v
veriweave-horizon-challenge
veriweave-vita-challenge
veriweave-boltzmann-challenge
```

## Outputs

| File | Contents |
|---|---|
| `result/manifest.json` | Environment, model, budgets, methods, and paper-validity checks |
| `result/traces.jsonl` | Per-task decision and verification traces |
| `result/metrics.csv` | Task-level metrics for analysis |
| `result/metrics.json` | Metrics in machine-readable form |
| `result/statistics.json` | Aggregate statistics |
| `result/failures.json` | Captured execution failures |
| `result/report.html` | Self-contained experiment report |
| `data/graph/evidence_graph.json` | Runtime-generated policy graph export |

Generated graph exports and experiment outputs are intentionally excluded from version control.

## Repository layout

```text
VeriWeave-VITA/
├── data/
│   ├── graph/schema.json
│   ├── policies/
│   └── tasks/
├── docs/
│   ├── architecture.md
│   └── benchmark_protocol.md
├── src/veriweave/
│   ├── provenance_robust_selection.py
│   ├── vita.py
│   ├── horizon.py
│   ├── argumentation.py
│   ├── temporal.py
│   ├── evaluators.py
│   └── main.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Reproducibility

- Freeze the model, prompts, seeds, graph construction, evaluator thresholds, and evidence budgets before a reported run.
- Keep `STRICT_MODEL_RUN=true` for scientific experiments.
- Retain the generated manifest, traces, failures, and metrics with every reported result.
- Do not mix offline fallback outputs with model-backed results.
- Report all methods under the same post-hoc audit and compute budget.

The detailed protocol is documented in [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

## Research scope and limitations

This repository is a research prototype, not a legal-compliance product, production authorization engine, or security boundary. The included policy corpus and benchmark are synthetic or real-world-inspired. PRO is deterministic under fixed inputs, but bounded retrieval and coalition search do not prove global completeness.

Strong empirical claims require independently curated policy corpora, multiple models and seeds, human-validated decisive evidence, faithful external baselines, and equal-compute comparisons.

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). GitHub can generate BibTeX and other formats from the repository's **Cite this repository** control.

## License

Licensed under the [Apache License 2.0](LICENSE).
