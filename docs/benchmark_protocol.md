# Benchmark protocol

The benchmark contains four balanced task families derived from expert-labeled policy cases:

- natural-language graph querying;
- hallucination and evidence verification;
- applicability and human-review routing;
- conflict and version-sensitive reasoning.

The default comparison includes Direct LLM, Text RAG, Graph RAG, and ClaimGraph. ClaimGraph adds atomic claim extraction and deterministic validation over a typed property graph.

Primary metrics are decision accuracy, review-routing accuracy, claim precision, unsupported-claim rate, contradiction detection, provenance coverage, temporal-validity accuracy, applicability accuracy, graph-reference recall, and traceability.

Offline fallback mode is intended only to verify the software pipeline. Its values must not be reported as scientific findings. Run a fixed LLM configuration over the complete benchmark and retain the generated manifest before using the results in a paper.
