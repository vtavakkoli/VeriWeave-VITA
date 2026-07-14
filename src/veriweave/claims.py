from __future__ import annotations

import json
import re

from .models import AtomicClaim
from .utils import split_sentences, stable_hash

_DECISION_TERMS = {
    "allowed", "not allowed", "conditional", "needs review", "human review",
    "may proceed", "may ", "can proceed", "is sufficient", "permitted", "must not", "prohibited", "requires", "required", "without review",
}


def extract_atomic_claims(answer: str) -> list[AtomicClaim]:
    """Deterministically decompose a response into auditable claim units.

    The extractor is deliberately transparent and stable across methods. It is
    used by the independent audit pass, rather than only by VeriWeave.
    """
    claims: list[AtomicClaim] = []
    for sentence in split_sentences(answer):
        clean = re.sub(r"^[-*\d.)\s]+", "", sentence).strip()
        if len(clean.split()) < 3:
            continue
        low = clean.lower()
        polarity = "negative" if any(term in low for term in ("must not", "not allowed", "prohibited", "cannot", "forbidden")) else "positive"
        decisive = any(term in low for term in _DECISION_TERMS)
        kind = "decision" if decisive else "assertion"
        claims.append(
            AtomicClaim(
                id=f"claim:{stable_hash(clean)}",
                text=clean,
                kind=kind,
                polarity=polarity,
                decisive=decisive,
            )
        )
    if not claims and answer.strip():
        text = answer.strip()
        claims.append(AtomicClaim(id=f"claim:{stable_hash(text)}", text=text))
    return claims


def parse_claims_from_model(raw: str) -> list[AtomicClaim]:
    try:
        match = re.search(r"(\{.*\})", raw, flags=re.S)
        obj = json.loads(match.group(1) if match else raw)
    except Exception:
        return extract_atomic_claims(raw)
    entries = obj.get("claims", []) if isinstance(obj, dict) else []
    parsed: list[AtomicClaim] = []
    for entry in entries:
        if isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
            kind = str(entry.get("kind", "assertion"))
            decisive = bool(entry.get("decisive", kind == "decision"))
        else:
            text = str(entry).strip()
            kind = "assertion"
            decisive = False
        if text:
            parsed.append(
                AtomicClaim(
                    id=f"claim:{stable_hash(text)}",
                    text=text,
                    kind=kind,
                    polarity="negative" if any(term in text.lower() for term in ("not", "prohibited", "cannot")) else "positive",
                    decisive=decisive,
                )
            )
    fallback_answer = str(obj.get("answer", "")) if isinstance(obj, dict) else raw
    return parsed or extract_atomic_claims(fallback_answer)
