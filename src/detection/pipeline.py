"""Detection pipeline — normalization (stage 1) + fast path (stage 2).

The proxy awaits :meth:`DetectionPipeline.process_request` on every chat
completion request before forwarding it upstream. Current stages:

- Stage 1 — ingress normalization (``src.normalization``): NFKC /
  zero-width / control-char cleanup, base64 / hex / ROT13 decode-and-recheck,
  pseudo-system delimiter neutralization. Findings are logged; depending on
  the configured ``mode`` the stage rewrites the payload (default), blocks
  with HTTP 403, or passes the original bytes through (log-only).
- Stage 2 — fast path regex/heuristics (``src.fast_path``): weighted
  noisy-OR scoring over known jailbreak patterns, run on the **normalized**
  text, so encoded or obfuscated inputs are seen the way the LLM would see
  them. score >= block threshold → HTTP 403; >= warn threshold → logged,
  still forwarded.

Later agents add more stages on top:

- Agent 4 — canary token detection
- Agent 5 — deep path ML classifier (ONNX)
- Agent 6 — pipeline orchestration (block / warn / log-only)

Note on stage interaction: stage 1 already neutralizes raw special tokens
(``<|im_start|>`` → ``⟦fake-im-start⟧``), so fast-path patterns aimed at raw
tokens mainly fire when the corresponding normalization toggles are disabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from ..config import FastPathSettings, NormalizationSettings
from ..fast_path import DetectionResult, FastPathDetector
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


def extract_user_text(payload: dict[str, Any]) -> str:
    """Concatenate the text of all user messages in a chat completion payload.

    OpenAI content may be a plain string or a list of typed parts
    (``[{"type": "text", "text": ...}, ...]``); both are handled. System and
    assistant messages are *not* scanned — they belong to the application,
    not the (untrusted) end user.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if (
                    isinstance(chunk, dict)
                    and chunk.get("type") == "text"
                    and isinstance(chunk.get("text"), str)
                ):
                    parts.append(chunk["text"])
    return "\n".join(parts)


class DetectionPipeline:
    """Runs stages over a chat completion payload: normalization, then fast path."""

    def __init__(
        self,
        normalization: NormalizationSettings | None = None,
        *,
        fast_path: FastPathSettings | None = None,
    ) -> None:
        self.normalization = normalization or NormalizationSettings()
        self.fast_path: FastPathDetector | None = None
        if fast_path is not None and fast_path.enabled:
            self.fast_path = FastPathDetector(
                block_threshold=fast_path.block_threshold,
                warn_threshold=fast_path.warn_threshold,
            )

    async def process_request(self, payload: dict[str, Any]) -> PipelineResult:
        """Inspect a chat completion payload and decide pass / block.

        Stage 1 normalizes every user-message content, rewrites the payload
        with the cleaned text, and logs the findings. Stage 2 runs the fast
        path over the *normalized* user text. Depending on the configured
        stage actions the result can block (403), rewrite, or pass through.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return PipelineResult()

        # ---------------------------------------------------------------
        # Stage 1 — ingress normalization (per user message, never raises)
        # ---------------------------------------------------------------
        findings: list[Finding] = []
        new_messages: list[Any] = []
        for message in messages:
            new_message, message_findings = self._normalize_message(message)
            new_messages.append(new_message)
            findings.extend(message_findings)

        if findings:
            self._log_findings(payload, findings)

        if self.normalization.mode == "block":
            categories = sorted({finding.category for finding in findings})
            return PipelineResult(
                action="block",
                reason=(
                    "Request blocked by normalization: " + ", ".join(categories)
                ),
            )

        result = PipelineResult()
        if findings and self.normalization.mode == "rewrite":
            new_payload = dict(payload)
            new_payload["messages"] = new_messages
            result = PipelineResult(payload=new_payload)

        # ---------------------------------------------------------------
        # Stage 2 — fast path, on the normalized text (what the LLM would
        # actually reason about after stage 1).
        # ---------------------------------------------------------------
        if self.fast_path is not None:
            text = extract_user_text({"messages": new_messages})
            if text.strip():
                detection = self.fast_path.detect(text)
                if detection.action == "block":
                    self._log_detection(detection, level="warning")
                    return PipelineResult(action="block", reason=detection.reason)
                if detection.action == "warn":
                    self._log_detection(detection, level="info")

        return result

    # ------------------------------------------------------------------
    # Stage 1 helpers (normalization)
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

    # ------------------------------------------------------------------
    # Stage 2 helpers (fast path)
    # ------------------------------------------------------------------

    def _log_detection(self, detection: DetectionResult, level: str) -> None:
        """Emit the structured fast-path detection event (block or warn)."""
        message = json.dumps(
            detection.log_dict(), ensure_ascii=False, sort_keys=True
        )
        if level == "warning":
            logger.warning("detection event: %s", message)
        else:
            logger.info("detection event: %s", message)
