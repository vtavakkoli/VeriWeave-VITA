from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .utils import split_sentences, stable_hash


@dataclass(frozen=True)
class AtomicClaim:
    id: str
    text: str
    kind: str = "assertion"
    polarity: str = "positive"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "kind": self.kind, "polarity": self.polarity, "metadata": self.metadata}


def extract_atomic_claims(answer: str) -> list[AtomicClaim]:
    """Split a generated response into verifiable atomic units.

    This deterministic extractor is intentionally transparent. An LLM extractor
    can be plugged in later, but the benchmark always retains this reproducible
    fallback.
    """
    claims: list[AtomicClaim] = []
    for sentence in split_sentences(answer):
        clean = re.sub(r"^[-*\d.)\s]+", "", sentence).strip()
        if len(clean.split()) < 3:
            continue
        low = clean.lower()
        polarity = "negative" if any(x in low for x in ["must not", "not allowed", "prohibited", "cannot"]) else "positive"
        kind = "decision" if any(x in low for x in ["allowed", "not allowed", "conditional", "review", "proceed"]) else "assertion"
        claims.append(AtomicClaim(f"claim:{stable_hash(clean)}", clean, kind, polarity))
    if not claims and answer.strip():
        claims.append(AtomicClaim(f"claim:{stable_hash(answer)}", answer.strip()))
    return claims


def parse_claims_from_model(raw: str) -> list[AtomicClaim]:
    try:
        match = re.search(r"(\{.*\})", raw, flags=re.S)
        obj = json.loads(match.group(1) if match else raw)
    except Exception:
        return extract_atomic_claims(raw)
    entries = obj.get("claims", []) if isinstance(obj, dict) else []
    out = []
    for entry in entries:
        text = str(entry.get("text", "")).strip() if isinstance(entry, dict) else str(entry).strip()
        if text:
            out.append(AtomicClaim(f"claim:{stable_hash(text)}", text, str(entry.get("kind", "assertion")) if isinstance(entry, dict) else "assertion"))
    return out or extract_atomic_claims(str(obj.get("answer", "")) if isinstance(obj, dict) else raw)
