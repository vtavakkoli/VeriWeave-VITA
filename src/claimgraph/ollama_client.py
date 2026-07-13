from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .utils import append_jsonl, now

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float, trace_path: Path, enabled: bool = True, healthcheck_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.trace_path = trace_path
        self.enabled = enabled
        self.healthcheck_seconds = healthcheck_seconds
        self._available: bool | None = None

    def is_available(self) -> bool:
        if not self.enabled:
            self._available = False
            return False
        if self._available is not None:
            return self._available
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=self.healthcheck_seconds) as response:
                response.read(128)
            self._available = True
        except Exception as exc:
            logger.warning("Ollama unavailable: %s; using deterministic fallback", exc)
            self._available = False
        return self._available

    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        if not self.is_available():
            append_jsonl(self.trace_path, {"timestamp": now(), "kind": "ollama_skipped", "meta": meta, "model": self.model})
            return ""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "seed": int(meta.get("seed", 7)), "num_ctx": 8192},
        }).encode()
        endpoint = f"{self.base_url}/api/generate"
        try:
            started = time.perf_counter()
            request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode()).get("response", "")
            append_jsonl(self.trace_path, {"timestamp": now(), "kind": "ollama", "meta": meta, "model": self.model, "latency_seconds": time.perf_counter() - started, "output": raw})
            return raw
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ollama request failed: %s; using deterministic fallback", exc)
            self._available = False
            append_jsonl(self.trace_path, {"timestamp": now(), "kind": "ollama_error", "meta": meta, "error": str(exc)})
            return ""
