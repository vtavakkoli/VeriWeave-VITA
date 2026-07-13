# ClaimGraph

**ClaimGraph** is a research prototype for provenance-aware validation of LLM-generated claims over enterprise policies and Web data.

> Paper direction: **ClaimGraph: Provenance-Aware Property-Graph Validation for Trustworthy LLM Reasoning over Enterprise Policies**

Instead of treating retrieved passages as an undifferentiated prompt, ClaimGraph represents policy documents, clauses, concepts, versions, authorities, and inter-clause relations in a typed property graph. It then decomposes an LLM answer into atomic claims and verifies each claim for support, contradiction, applicability, temporal validity, and provenance completeness.

## Research distinction

```text
Text RAG
request -> retrieve passages -> generate answer

ClaimGraph
request -> graph retrieval -> generate candidate answer
        -> extract atomic claims
        -> validate support / conflict / applicability / version / provenance
        -> produce claim-evidence subgraph
        -> route unresolved claims to human review
```

Its central object is an atomic claim connected to a verifiable evidence subgraph.

## Methods

1. **Direct LLM** — closed-book generation.
2. **Text RAG** — BM25-style retrieval over policy clauses.
3. **Graph RAG** — graph-aware clause retrieval with concept and version paths.
4. **ClaimGraph** — Graph RAG plus atomic-claim validation and review routing.

## Property-graph schema

Node types:

- `Document`
- `Clause`
- `Concept`
- `Version`
- `Authority`

Edge types:

- `DERIVED_FROM`
- `VALID_IN`
- `APPLIES_TO`
- `SUPPORTS`
- `CONTRADICTS`
- `OVERRIDES`
- `GOVERNED_BY`

The graph is exported to:

```text
data/graph/property_graph.json
```

## Outputs

```text
result/report.html
result/metrics.csv
result/metrics.json
result/traces.jsonl
result/failures.json
result/statistics.json
result/manifest.json
```

Each ClaimGraph trace includes the generated answer, atomic claims, per-claim validation results, citations, retrieved evidence, and a compact claim-evidence subgraph.

## Repository structure

```text
ClaimGraph/
  data/
    policies/                    policy corpus
    tasks/claimgraph_benchmark.jsonl
    graph/schema.json
  docs/
    architecture.md
    benchmark_protocol.md
  src/claimgraph/
    graph.py                     typed property graph and JSON export
    retrieval.py                 text and graph retrieval
    claims.py                    atomic claim extraction
    validators.py                support/conflict/version/provenance checks
    agents.py                    four comparison methods
    evaluators.py                trustworthiness metrics
    report_generator.py          self-contained HTML report
    main.py                      experiment runner
  tests/
```

## Run without an LLM

The deterministic fallback checks the entire software pipeline without making scientific performance claims:

```bash
OLLAMA_ENABLED=false MAX_TASKS=8 docker compose up --build
```

Open:

```text
result/report.html
```

## Run with Ollama

Install Ollama on the host, pull a model, and start the service:

```bash
ollama pull gemma4:e2b
ollama serve
```

Then run all 600 expert-labeled tasks:

```bash
docker compose up --build
```

For local execution:

```bash
PYTHONPATH=src OLLAMA_ENABLED=false MAX_TASKS=8 python -m claimgraph.main
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Metrics

| Metric | Purpose |
|---|---|
| `claim_precision` | Share of generated claims supported by graph evidence |
| `unsupported_claim_rate` | Share of contradicted or unresolved claims |
| `contradiction_detection` | Whether conflicts trigger safe handling |
| `provenance_coverage` | Whether support includes source, version, and citation |
| `temporal_validity_accuracy` | Whether version-sensitive claims use current evidence |
| `applicability_accuracy` | Connection between claims and applicable graph concepts |
| `graph_reference_recall` | Recovery of expected policy sections |
| `review_routing_accuracy` | Correct escalation of unresolved or high-impact cases |
| `traceability_score` | Availability of evidence, citations, and explanation subgraph |

## Research-use warning

The policy corpus is synthetic or real-world-inspired and de-identified. The benchmark labels are suitable for reproducible prototype evaluation, but external validity requires evaluation with an independent enterprise or public-sector corpus. Results produced in offline fallback mode must not be presented as LLM benchmark results.
