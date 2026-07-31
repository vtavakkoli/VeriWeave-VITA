from __future__ import annotations

import json
import re
import time
from typing import Any

from .data_loader import BenchmarkTask
from .envelope import build_verification_envelope
from .graph import PropertyGraph
from .models import CandidateResponse, VALID_DECISIONS
from .ollama_client import OllamaClient
from .retrieval import (
    CommunityGraphRetriever,
    Evidence,
    PPRGraphRetriever,
    SteinerGraphRetriever,
    TextRetriever,
    VeriWeaveRetriever,
)
from .utils import first_sentence, now


def _decision_from_answer_only(answer: str) -> str:
    lower = answer.lower()
    if any(term in lower for term in ("must not", "not allowed", "prohibited", "forbidden")):
        return "not_allowed"
    if any(term in lower for term in ("human review", "requires review", "needs review", "unresolved conflict")):
        return "needs_review"
    if any(term in lower for term in ("only if", "conditional", "requires", "subject to")):
        return "conditional"
    if any(term in lower for term in ("allowed", "permitted", "may proceed")):
        return "allowed"
    return "unknown"


def _fallback_candidate(evidence: list[Evidence]) -> CandidateResponse:
    if evidence:
        best = evidence[0]
        answer = f"{first_sentence(best.text)} [{best.citation_id}]"
        return CandidateResponse(
            answer=answer,
            decision=_decision_from_answer_only(answer),
            citations=[best.citation_id],
            human_review_required=False,
            raw="",
            parse_status="offline_fallback",
        )
    answer = "The available evidence is insufficient to verify this request. Human review is required."
    return CandidateResponse(
        answer=answer,
        decision="needs_review",
        citations=[],
        human_review_required=True,
        raw="",
        parse_status="offline_fallback",
    )


def _parse_candidate(raw: str, evidence: list[Evidence]) -> CandidateResponse:
    if not raw:
        return _fallback_candidate(evidence)
    try:
        match = re.search(r"(\{.*\})", raw, flags=re.S)
        obj = json.loads(match.group(1) if match else raw)
        answer = str(obj.get("answer", "")).strip()
        decision = str(obj.get("decision", "unknown")).strip().lower()
        citations_raw = obj.get("citations", [])
        citations = [str(value).strip() for value in citations_raw] if isinstance(citations_raw, list) else []
        review = bool(obj.get("human_review_required", False))
        if not answer:
            return _fallback_candidate(evidence)
        if decision not in VALID_DECISIONS:
            decision = "unknown"
        return CandidateResponse(answer, decision, citations, review, raw, "parsed")
    except Exception:
        answer = first_sentence(raw) or _fallback_candidate(evidence).answer
        return CandidateResponse(
            answer=answer,
            decision=_decision_from_answer_only(answer),
            citations=[],
            human_review_required=_decision_from_answer_only(answer) in {"unknown", "needs_review"},
            raw=raw,
            parse_status="unstructured_recovery",
        )


def _prompt(task: BenchmarkTask, evidence: list[Evidence], method: str) -> str:
    evidence_block = "\n".join(
        f"[{item.citation_id}] role={item.role}; version={item.version}; current={item.is_current}; modality={item.modality}; text={item.text}"
        for item in evidence
    )
    if method.startswith("VeriWeave-"):
        method_guidance = (
            "Use support and counterevidence, respect current versions and precedence, "
            "write one atomic sentence per claim, and cite every sentence with at least "
            "one evidence identifier. Mark needs_review only when the supplied evidence "
            "is genuinely insufficient or conflicting."
        )
        generation_profile = "VeriWeave"
    else:
        method_guidance = {
            "Direct LLM": "Answer without external evidence. Mark unknown or needs_review when uncertain.",
            "Text RAG": "Use only the retrieved text passages.",
            "Community GraphRAG": "Use the concept-community evidence and its graph paths.",
            "PPR GraphRAG": "Use the associatively ranked graph evidence.",
            "Steiner GraphRAG": "Use the connected-subgraph evidence.",
        }[method]
        generation_profile = method
    return f"""You are participating in a controlled policy question-answering benchmark.
{method_guidance}
Return exactly one JSON object with these keys:
- answer: concise answer written as atomic factual or normative sentences
- decision: allowed | not_allowed | conditional | needs_review | unknown
- citations: array containing only evidence identifiers shown below
- human_review_required: boolean
Do not copy a decision from retrieved evidence unless it answers the question. Do not invent citations.

Generation profile: {generation_profile}
Question: {task.question}
Evidence:
{evidence_block or '[none]'}
"""


def _generate_candidate(task: BenchmarkTask, evidence: list[Evidence], client: OllamaClient, method: str) -> CandidateResponse:
    raw = client.generate(_prompt(task, evidence, method), {"method": method, "task_id": task.id, "expect_json": True})
    return _parse_candidate(raw, evidence)


def _retriever_for(
    method: str,
    text_retriever: TextRetriever,
    community_retriever: CommunityGraphRetriever,
    ppr_retriever: PPRGraphRetriever,
    steiner_retriever: SteinerGraphRetriever,
    veriweave_retriever: VeriWeaveRetriever,
):
    return {
        "Text RAG": text_retriever,
        "Community GraphRAG": community_retriever,
        "PPR GraphRAG": ppr_retriever,
        "Steiner GraphRAG": steiner_retriever,
        "VeriWeave-Core": veriweave_retriever,
        "VeriWeave-Horizon": veriweave_retriever,
        "VeriWeave-VITA": veriweave_retriever,
        "VeriWeave-VITA-BPA": veriweave_retriever,
        "VeriWeave-VITA-PRO": veriweave_retriever,
    }.get(method)


def run_method(
    method: str,
    task: BenchmarkTask,
    graph: PropertyGraph,
    text_retriever: TextRetriever,
    community_retriever: CommunityGraphRetriever,
    ppr_retriever: PPRGraphRetriever,
    steiner_retriever: SteinerGraphRetriever,
    veriweave_retriever: VeriWeaveRetriever,
    client: OllamaClient,
    top_k: int,
    vita_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    retriever = _retriever_for(method, text_retriever, community_retriever, ppr_retriever, steiner_retriever, veriweave_retriever)
    evidence = [] if method == "Direct LLM" else retriever.retrieve(task.question, top_k)
    candidate = _generate_candidate(task, evidence, client, method)

    envelope = None
    answer = candidate.answer
    decision = candidate.decision
    human_review_required = candidate.human_review_required
    citations = candidate.citations
    claims: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    output_evidence = [item.to_dict() for item in evidence]

    if method in {"VeriWeave-Core", "VeriWeave-Horizon", "VeriWeave-VITA", "VeriWeave-VITA-BPA", "VeriWeave-VITA-PRO"}:
        envelope = build_verification_envelope(
            candidate,
            evidence,
            graph,
            question=task.question,
            enable_horizon=method == "VeriWeave-Horizon",
            enable_vita=method in {"VeriWeave-VITA", "VeriWeave-VITA-BPA", "VeriWeave-VITA-PRO"},
            enable_boltzmann_attention=method == "VeriWeave-VITA-BPA",
            enable_provenance_robust_selection=method == "VeriWeave-VITA-PRO",
            repair_verified_citations=method == "VeriWeave-VITA-PRO",
            vita_options=vita_options,
        )
        answer = envelope.verified_answer
        decision = envelope.gated_decision
        human_review_required = envelope.human_review_required
        citations = envelope.accepted_citations
        claims = envelope.claims
        validations = envelope.validations
        resolutions = envelope.resolutions
        output_evidence = envelope.effective_evidence

    subgraph_clause_ids = [
        str(item.get("clause_id", "")) for item in output_evidence[: max(3, top_k)] if item.get("clause_id")
    ]
    subgraph = (
        graph.explanation_subgraph(subgraph_clause_ids, include_claims=claims)
        if method in {"Community GraphRAG", "PPR GraphRAG", "Steiner GraphRAG", "VeriWeave-Core", "VeriWeave-Horizon", "VeriWeave-VITA", "VeriWeave-VITA-BPA", "VeriWeave-VITA-PRO"}
        else {"nodes": [], "edges": []}
    )

    return {
        "timestamp": now(),
        "method": method,
        "task_id": task.id,
        "task_type": task.task_type,
        "question": task.question,
        "candidate": candidate.to_dict(),
        "answer": answer,
        "decision": decision,
        "human_review_required": human_review_required,
        "claims": claims,
        "method_validations": validations,
        "method_resolutions": resolutions,
        "citations": citations,
        "initial_evidence": [item.to_dict() for item in evidence],
        "evidence": output_evidence,
        "claim_evidence_subgraph": subgraph,
        "verification_envelope": envelope.to_dict() if envelope else None,
        "evidence_horizon_certificate": envelope.evidence_horizon if envelope else None,
        "vita_certificate": envelope.vita_certificate if envelope else None,
        "argumentation_certificate": envelope.argumentation_certificate if envelope else None,
        "temporal_drift_certificate": envelope.temporal_drift_certificate if envelope else None,
        "boltzmann_policy_attention_certificate": envelope.boltzmann_policy_attention if envelope else None,
        "provenance_robust_selection_certificate": (
            (envelope.vita_certificate or {}).get("selection_certificate") if envelope else None
        ),
        "latency_seconds": round(time.perf_counter() - started, 6),
    }
