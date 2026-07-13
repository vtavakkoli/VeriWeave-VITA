from __future__ import annotations

import json
import re
import time
from typing import Any

from .claims import extract_atomic_claims
from .data_loader import BenchmarkTask
from .graph import PropertyGraph
from .ollama_client import OllamaClient
from .retrieval import Evidence, GraphRetriever, TextRetriever
from .utils import first_sentence, now, tokens
from .validators import validate_claims

VALID_DECISIONS = {"allowed", "not_allowed", "conditional", "needs_review", "unknown"}


def _decision_from_text(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ["must not", "not allowed", "prohibited", "forbidden"]):
        return "not_allowed"
    if any(x in low for x in ["human review", "requires review", "policy owner", "unresolved", "conflict"]):
        return "needs_review"
    if any(x in low for x in ["requires", "must", "only if", "conditional", "safeguard"]):
        return "conditional"
    if any(x in low for x in ["allowed", "may proceed", "permitted"]):
        return "allowed"
    return "unknown"


def _review_required(question: str, validations: list[dict], decision: str) -> bool:
    q = question.lower()
    high_impact = any(x in q for x in ["adverse", "eligibility", "benefit", "employee", "citizen", "high-risk", "sensitive"])
    unresolved = any(v["status"] != "supported" for v in validations)
    return high_impact or unresolved or decision in {"needs_review", "unknown"}


def _fallback_answer(task: BenchmarkTask, evidence: list[Evidence]) -> str:
    if evidence:
        best = evidence[0]
        return f"{first_sentence(best.text)} [{best.citation_id}]"
    return "The available evidence is insufficient to verify this request. Human review is required."


def _model_answer(task: BenchmarkTask, evidence: list[Evidence], client: OllamaClient, method: str) -> str:
    evidence_block = "\n".join(f"[{item.citation_id}] {item.text}" for item in evidence)
    prompt = f"""You are a policy-verification assistant. Answer the question using concise atomic claims.
Return JSON with keys answer, decision, citations, and human_review_required.
Allowed decisions: allowed, not_allowed, conditional, needs_review, unknown.
Do not invent evidence. Cite only identifiers shown below.

Question: {task.question}
Evidence:
{evidence_block or '[none]'}
"""
    raw = client.generate(prompt, {"method": method, "task_id": task.id, "expect_json": True})
    if not raw:
        return _fallback_answer(task, evidence)
    try:
        match = re.search(r"(\{.*\})", raw, re.S)
        obj = json.loads(match.group(1) if match else raw)
        return str(obj.get("answer", "")).strip() or _fallback_answer(task, evidence)
    except Exception:
        return first_sentence(raw) or _fallback_answer(task, evidence)


def _trace(method: str, task: BenchmarkTask, answer: str, evidence: list[Evidence], graph: PropertyGraph, started: float, verify: bool) -> dict[str, Any]:
    claims = extract_atomic_claims(answer)
    validations = [v.to_dict() for v in validate_claims(claims, evidence, graph)] if verify else []
    decision_source = " ".join([answer] + [item.text for item in evidence[:2]])
    decision = _decision_from_text(decision_source)
    if verify and any(v["status"] == "contradicted" for v in validations):
        decision = "needs_review"
    review = _review_required(task.question, validations, decision) if verify else decision in {"needs_review", "unknown"}
    cited = [item.citation_id for item in evidence if item.citation_id in answer]
    if verify and not cited:
        cited = [item.citation_id for item in evidence[:2]]
    subgraph = graph.explanation_subgraph([item.clause_id for item in evidence[:3]]) if method in {"Graph RAG", "ClaimGraph"} else {"nodes": [], "edges": []}
    return {
        "timestamp": now(),
        "method": method,
        "task_id": task.id,
        "task_type": task.task_type,
        "question": task.question,
        "answer": answer,
        "decision": decision,
        "human_review_required": review,
        "claims": [claim.to_dict() for claim in claims],
        "validations": validations,
        "citations": cited,
        "evidence": [item.to_dict() for item in evidence],
        "claim_evidence_subgraph": subgraph,
        "latency_seconds": round(time.perf_counter() - started, 6),
    }


def run_method(method: str, task: BenchmarkTask, graph: PropertyGraph, text_retriever: TextRetriever, graph_retriever: GraphRetriever, client: OllamaClient, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    if method == "Direct LLM":
        evidence: list[Evidence] = []
        answer = _model_answer(task, evidence, client, method)
        return _trace(method, task, answer, evidence, graph, started, verify=False)
    if method == "Text RAG":
        evidence = text_retriever.retrieve(task.question, top_k)
        answer = _model_answer(task, evidence, client, method)
        return _trace(method, task, answer, evidence, graph, started, verify=False)
    if method == "Graph RAG":
        evidence = graph_retriever.retrieve(task.question, top_k)
        answer = _model_answer(task, evidence, client, method)
        return _trace(method, task, answer, evidence, graph, started, verify=False)
    if method == "ClaimGraph":
        evidence = graph_retriever.retrieve(task.question, top_k)
        answer = _model_answer(task, evidence, client, method)
        return _trace(method, task, answer, evidence, graph, started, verify=True)
    raise ValueError(f"Unknown method: {method}")
