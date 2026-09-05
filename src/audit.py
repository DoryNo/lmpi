"""Structured JSONL audit log + optional access log (Agent 10, hardening v1.1).

One JSON object per line, written to a configurable sink:

- ``LMPI_AUDIT_PATH=stdout`` / ``stderr`` — the corresponding process stream
  (handy for Docker log collection).
- ``LMPI_AUDIT_PATH=/path/to/audit.jsonl`` — a file opened in append mode,
  flushed on every line and closed on graceful shutdown.
- ``LMPI_AUDIT_ENABLED=false`` (the default) — no audit sink at all.

Security invariants (enforced by the writers, not the sink):

- Canary token **values** are never written — only HMAC fingerprints.
- Prompt text is excluded by default (``LMPI_AUDIT_INCLUDE_TEXT=false``);
  when enabled, the caller scrubs any canary occurrence first.

The :class:`AccessLogMiddleware` is an optional access-log style entry per
HTTP request (method, path, status, duration, client key). Client keys are
hashed — raw ``Authorization``/``x-api-key`` values are never logged, and no
request/response bodies are recorded.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import AUDIT_STREAM_TARGETS, AuditSettings

# Headers that may carry a client credential, by raw header name.
_CLIENT_KEY_HEADERS = ("authorization", "x-api-key")


def now_iso() -> str:
    """UTC timestamp for audit events (``2026-01-01T00:00:00.000000+00:00``)."""
    return datetime.now(timezone.utc).isoformat()


def redact_client_key(value: str | None) -> str:
    """Hash a client credential down to a stable, non-reversible key.

    ``"sha256:<8 hex>"`` — enough to correlate requests from one client
    without leaking the token itself.
    """
    if not value:
        return "none"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"sha256:{digest}"


def client_key_from_headers(headers: Any) -> str:
    """Redacted client key from raw ASGI/Starlette-style headers."""
    items: dict[str, str] = {}
    if hasattr(headers, "items"):
        try:
            items = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in headers.items()
            }
        except (AttributeError, UnicodeDecodeError):
            items = {}
    else:
        try:
            for key, value in headers:
                items[key.decode("latin-1").lower()] = value.decode("latin-1")
        except (TypeError, AttributeError, UnicodeDecodeError):
            items = {}
    for name in _CLIENT_KEY_HEADERS:
        if items.get(name):
            return redact_client_key(items[name])
    return "none"


class AuditSink:
    """JSON-lines audit writer with a pluggable target and safe close."""

    def __init__(self, settings: AuditSettings) -> None:
        self.settings = settings
        self.include_text = settings.include_text
        self._lock = threading.Lock()
        self._file = None
        self._stream: Any = None
        target = settings.path
        if target == "stdout":
            self._stream = sys.stdout
        elif target == "stderr":
            self._stream = sys.stderr
        else:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered text mode: every JSON line lands on disk when
            # written, so a crash loses nothing already recorded.
            self._file = open(path, "a", encoding="utf-8", buffering=1)

    @property
    def closed(self) -> bool:
        return self._file is not None and self._file.closed

    def record(self, event: dict[str, Any]) -> None:
        """Write one JSON object as a single line. Never raises to callers."""
        event = {**event, "ts": event.get("ts") or now_iso()}
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            if self._file is not None:
                try:
                    self._file.write(line + "\n")
                except OSError:
                    pass
            elif self._stream is not None:
                try:
                    self._stream.write(line + "\n")
                    self._stream.flush()
                except (OSError, ValueError):
                    pass

    def close(self) -> None:
        """Flush and close the file sink; stream sinks are only flushed."""
        with self._lock:
            if self._file is not None and not self._file.closed:
                try:
                    self._file.flush()
                    self._file.close()
                except OSError:
                    pass


def build_audit_sink(settings: AuditSettings) -> AuditSink | None:
    """Audit sink for the app, or ``None`` when auditing is disabled."""
    if not settings.enabled:
        return None
    return AuditSink(settings)


class AccessLogMiddleware:
    """Pure ASGI middleware emitting one access-log event per HTTP request.

    Records method, path, status, duration and a *hashed* client key. No
    bodies, no headers — safe by construction.
    """

    def __init__(self, app: Any, sink: AuditSink) -> None:
        self.app = app
        self.sink = sink

    async def __call__(self, scope: Any, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status: int | None = None

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self.sink.record(
                {
                    "event": "access",
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "status": status,
                    "duration_ms": round(duration_ms, 2),
                    "client_key": client_key_from_headers(scope.get("headers")),
                }
            )
