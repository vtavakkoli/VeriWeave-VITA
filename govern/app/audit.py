from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models import AuditEnvelope


class AuditLedger:
    """Append-only JSONL audit chain with optional HMAC signatures."""

    def __init__(self, path: Path, signing_key: str | None = None) -> None:
        self.path = path
        self.signing_key = signing_key.encode("utf-8") if signing_key else None
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _last_hash(self) -> str:
        last = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)["record_hash"]
        return last or "0" * 64

    def append(self, payload: dict[str, Any]) -> AuditEnvelope:
        with self._lock:
            previous_hash = self._last_hash()
            created_at = datetime.now(timezone.utc)
            record_id = str(uuid4())
            body = {
                "record_id": record_id,
                "created_at": created_at.isoformat(),
                "previous_hash": previous_hash,
                "payload": payload,
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
            record_hash = hashlib.sha256((previous_hash + canonical).encode("utf-8")).hexdigest()
            signature = (
                hmac.new(self.signing_key, record_hash.encode("utf-8"), hashlib.sha256).hexdigest()
                if self.signing_key
                else None
            )
            record = {**body, "record_hash": record_hash, "signature": signature}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

            return AuditEnvelope(
                record_id=record_id,
                created_at=created_at,
                record_hash=record_hash,
                previous_hash=previous_hash,
                signature=signature,
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return list(reversed(rows[-limit:]))

    def verify(self) -> dict[str, Any]:
        previous_hash = "0" * 64
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                body = {
                    "record_id": record["record_id"],
                    "created_at": record["created_at"],
                    "previous_hash": record["previous_hash"],
                    "payload": record["payload"],
                }
                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
                expected_hash = hashlib.sha256(
                    (previous_hash + canonical).encode("utf-8")
                ).hexdigest()
                if record["previous_hash"] != previous_hash or record["record_hash"] != expected_hash:
                    return {"valid": False, "records": count, "failed_at_line": line_number}
                previous_hash = record["record_hash"]
                count += 1
        return {"valid": True, "records": count, "head_hash": previous_hash}
