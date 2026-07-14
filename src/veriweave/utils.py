from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "if", "in",
    "is", "it", "may", "must", "of", "on", "or", "should", "that", "the", "this", "to", "with",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def token_list(text: str, *, remove_stopwords: bool = False) -> list[str]:
    values = [token.lower() for token in TOKEN_RE.findall(text or "")]
    return [token for token in values if token not in STOPWORDS] if remove_stopwords else values


def tokens(text: str, *, remove_stopwords: bool = True) -> set[str]:
    return set(token_list(text, remove_stopwords=remove_stopwords))


def token_similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def containment(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta))


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:n]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\s*;\s*", clean) if part.strip()]


def first_sentence(text: str, max_chars: int = 300) -> str:
    parts = split_sentences(text)
    return (parts[0] if parts else "")[:max_chars].rstrip()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
