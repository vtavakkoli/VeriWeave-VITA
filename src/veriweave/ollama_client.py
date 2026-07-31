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


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["allowed", "not_allowed", "conditional", "needs_review", "unknown"],
        },
        "citations": {"type": "array", "items": {"type": "string"}},
        "human_review_required": {"type": "boolean"},
    },
    "required": ["answer", "decision", "citations", "human_review_required"],
}


class OllamaClient:
    """Small resilient Ollama client for reproducible benchmark runs.

    A transient 5xx response no longer disables the model for the rest of the
    experiment.  Requests are retried with bounded exponential backoff and a
    new health check.  In strict mode, exhausted retries stop the run instead of
    silently mixing model outputs with deterministic fallback outputs.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        trace_path: Path,
        enabled: bool = True,
        healthcheck_seconds: float = 2.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        strict_model_run: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        num_ctx: int = 8192,
        qwen_no_think: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.trace_path = trace_path
        self.enabled = enabled
        self.healthcheck_seconds = healthcheck_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.strict_model_run = strict_model_run
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.qwen_no_think = qwen_no_think
        self._available: bool | None = None
        self.call_stats = {
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "fallback": 0,
            "healthcheck_failures": 0,
        }

    def is_available(self, *, force: bool = False) -> bool:
        if not self.enabled:
            self._available = False
            return False
        if self._available is True and not force:
            return True
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=self.healthcheck_seconds) as response:
                response.read(128)
            self._available = True
        except Exception as exc:
            self.call_stats["healthcheck_failures"] += 1
            logger.warning("Ollama health check failed: %s", exc)
            self._available = None  # permit a later retry
            return False
        return True

    def _fallback_or_raise(self, meta: dict[str, Any], message: str) -> str:
        self.call_stats["failed"] += 1
        if self.enabled and self.strict_model_run:
            raise RuntimeError(message)
        self.call_stats["fallback"] += 1
        append_jsonl(
            self.trace_path,
            {
                "timestamp": now(),
                "kind": "ollama_fallback",
                "meta": meta,
                "model": self.model,
                "reason": message,
            },
        )
        return ""

    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        if self.qwen_no_think and "qwen3" in self.model.lower() and "/no_think" not in prompt:
            prompt = prompt.rstrip() + "\n/no_think"
        if not self.enabled:
            self.call_stats["fallback"] += 1
            append_jsonl(
                self.trace_path,
                {"timestamp": now(), "kind": "ollama_skipped", "meta": meta, "model": self.model},
            )
            return ""
        if not self.is_available():
            return self._fallback_or_raise(meta, "Ollama is unavailable before generation.")

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": _RESPONSE_SCHEMA,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "seed": int(meta.get("seed", 7)),
                    "num_ctx": self.num_ctx,
                },
            }
        ).encode()
        endpoint = f"{self.base_url}/api/generate"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                started = time.perf_counter()
                request = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode()).get("response", "")
                self._available = True
                self.call_stats["successful"] += 1
                append_jsonl(
                    self.trace_path,
                    {
                        "timestamp": now(),
                        "kind": "ollama",
                        "meta": meta,
                        "model": self.model,
                        "attempt": attempt + 1,
                        "latency_seconds": time.perf_counter() - started,
                        "output": raw,
                    },
                )
                return raw
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                self._available = None
                append_jsonl(
                    self.trace_path,
                    {
                        "timestamp": now(),
                        "kind": "ollama_error",
                        "meta": meta,
                        "model": self.model,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )
                if attempt >= self.max_retries:
                    break
                self.call_stats["retried"] += 1
                delay = self.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Ollama request failed (attempt %s/%s): %s; retrying in %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                if delay:
                    time.sleep(delay)
                self.is_available(force=True)

        return self._fallback_or_raise(
            meta,
            f"Ollama request failed after {self.max_retries + 1} attempt(s): {last_error}",
        )
