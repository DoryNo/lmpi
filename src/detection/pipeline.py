"""Detection pipeline — normalization is stage 1.

The proxy awaits :meth:`DetectionPipeline.process_request` on every chat
completion request before forwarding it upstream. Current stages:

- Stage 1 — ingress normalization (this module + ``src.normalization``):
  NFKC / zero-width / control-char cleanup, base64 / hex / ROT13
  decode-and-recheck, pseudo-system delimiter neutralization.

Later agents add more stages on top:

- Agent 3 — fast path regex/heuristics (runs on the normalized text)
- Agent 4 — canary token detection
- Agent 5 — deep path ML classifier (ONNX)
- Agent 6 — pipeline orchestration (block / warn / log-only)

Normalization alone never blocks by default (``mode: "rewrite"``): findings
are logged and the payload is rewritten with the cleaned text. Blocking,
when explicitly configured (``mode: "block"``), yields HTTP 403.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from ..config import NormalizationSettings
from ..normalization import normalize
from ..normalization.types import Finding

logger = logging.getLogger("lmpi.detection")

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
    """Runs stages over a chat completion payload; normalization first."""

    def __init__(
        self, normalization: NormalizationSettings | None = None
    ) -> None:
        self.normalization = normalization or NormalizationSettings()

    async def process_request(self, payload: dict[str, Any]) -> PipelineResult:
        """Inspect a chat completion payload.

        Stage 1 normalizes every user-message content, rewrites the payload
        with the cleaned text, and logs the findings. Depending on the
        configured normalization mode the stage can also block (403) or
        pass the original payload through untouched (log-only).
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return PipelineResult()

        findings: list[Finding] = []
        new_messages: list[Any] = []
        for message in messages:
            new_message, message_findings = self._normalize_message(message)
            new_messages.append(new_message)
            findings.extend(message_findings)

        if findings:
            self._log_findings(payload, findings)
        else:
            return PipelineResult()

        if self.normalization.mode == "block":
            categories = sorted({finding.category for finding in findings})
            return PipelineResult(
                action="block",
                reason=(
                    "Request blocked by normalization: " + ", ".join(categories)
                ),
            )

        if self.normalization.mode != "rewrite":
            # "log" (and any future read-only mode): keep original bytes.
            return PipelineResult()

        new_payload = dict(payload)
        new_payload["messages"] = new_messages
        return PipelineResult(payload=new_payload)

    # ------------------------------------------------------------------
    # Message-level helpers
    # ------------------------------------------------------------------

    def _normalize_message(
        self, message: Any
    ) -> tuple[Any, list[Finding]]:
        """Normalize one chat message (user role only). Never raises."""
        if not isinstance(message, dict) or message.get("role") != "user":
            return message, []
        content = message.get("content")
        if isinstance(content, str):
            result = normalize(
                content,
                unicode_cleaning=self.normalization.unicode,
                base64=self.normalization.base64,
                hex=self.normalization.hex,
                rot13=self.normalization.rot13,
                delimiters=self.normalization.delimiters,
            )
            if not result.changed:
                return message, []
            new_message = dict(message)
            new_message["content"] = result.cleaned_text
            return new_message, result.findings
        if isinstance(content, list):
            # OpenAI-style multipart content: normalize each text part only.
            new_parts: list[Any] = []
            findings: list[Finding] = []
            changed = False
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    result = normalize(
                        part["text"],
                        unicode_cleaning=self.normalization.unicode,
                        base64=self.normalization.base64,
                        hex=self.normalization.hex,
                        rot13=self.normalization.rot13,
                        delimiters=self.normalization.delimiters,
                    )
                    findings.extend(result.findings)
                    if result.changed:
                        changed = True
                        part = {**part, "text": result.cleaned_text}
                new_parts.append(part)
            if not changed:
                return message, []
            new_message = dict(message)
            new_message["content"] = new_parts
            return new_message, findings
        return message, []

    def _log_findings(
        self, payload: dict[str, Any], findings: list[Finding]
    ) -> None:
        """Emit one structured JSON log line per request with findings."""
        event = {
            "stage": "normalization",
            "mode": self.normalization.mode,
            "model": payload.get("model"),
            "findings": [finding.to_dict() for finding in findings],
        }
        logger.info("detection event: %s", json.dumps(event, ensure_ascii=False))
