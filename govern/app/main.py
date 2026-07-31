from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.audit import AuditLedger
from app.engine import GovernanceEngine
from app.evidence import EvidenceVerifier
from app.models import EvaluationRequest, EvaluationResponse
from app.policy import PolicyLoadError, PolicyStore

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = Path(os.getenv("GOVERN_POLICY_DIR", BASE_DIR / "config" / "policies"))
AUDIT_PATH = Path(os.getenv("GOVERN_AUDIT_PATH", BASE_DIR / "data" / "audit.jsonl"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

policy_store = PolicyStore(POLICY_DIR)
audit_ledger = AuditLedger(AUDIT_PATH, os.getenv("GOVERN_AUDIT_SIGNING_KEY"))
engine = GovernanceEngine(policy_store, EvidenceVerifier(), audit_ledger)

app = FastAPI(
    title="VeriWeave Govern",
    version=__version__,
    description=(
        "Policy enforcement and evidence governance control plane for enterprise AI agents. "
        "Every decision is policy-versioned, evidence-assessed, review-routed and audit-chained."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "veriweave-govern",
        "version": __version__,
        "active_policies": len(policy_store.policies),
        "policy_set_hash": policy_store.policy_set_hash,
    }


@app.get("/v1/policies")
def list_policies() -> dict[str, object]:
    return {"policy_set_hash": policy_store.policy_set_hash, "policies": policy_store.describe()}


@app.post("/v1/policies/reload")
def reload_policies() -> dict[str, object]:
    try:
        policy_store.reload()
    except PolicyLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return list_policies()


@app.post("/v1/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    return engine.evaluate(request)


@app.get("/v1/audit")
def recent_audit(limit: int = Query(default=25, ge=1, le=500)) -> dict[str, object]:
    return {"records": audit_ledger.recent(limit), "integrity": audit_ledger.verify()}


@app.get("/v1/audit/verify")
def verify_audit() -> dict[str, object]:
    return audit_ledger.verify()


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("GOVERN_HOST", "0.0.0.0"),
        port=int(os.getenv("GOVERN_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    run()
