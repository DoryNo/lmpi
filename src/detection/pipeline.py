"""No-op detection pipeline placeholder.

The proxy awaits :meth:`DetectionPipeline.process_request` on every chat
completion request before forwarding it upstream. Later agents replace this
no-op with real stages:

- Agent 2 — ingress normalization (NFKC, zero-width, base64/hex/rot13)
- Agent 3 — fast path regex/heuristics
- Agent 4 — canary token detection
- Agent 5 — deep path ML classifier (ONNX)
- Agent 6 — pipeline orchestration (block / warn / log-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Action = Literal["pass", "block"]


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of running a request through the detection pipeline.

    Attributes:
        action: ``"pass"`` forwards the request, ``"block"`` rejects it
            with HTTP 403.
        reason: Human-readable explanation, returned to the client when
            blocking and written to the log.
        payload: Replacement request payload. ``None`` keeps the original
            request body bytes untouched (byte-for-byte transparency).
    """

    action: Action = "pass"
    reason: str | None = None
    payload: dict[str, Any] | None = None


class DetectionPipeline:
    """Pass-through pipeline hook — every request is forwarded unchanged."""

    async def process_request(self, payload: dict[str, Any]) -> PipelineResult:
        """Inspect a chat completion payload. No-op: always passes."""
        return PipelineResult()
