"""Detection pipeline — normalization (stage 1) + fast path (stage 2) + deep path (stage 3).

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
- Stage 3 — deep path ML classifier (``src.deep_path``): ONNX DeBERTa
  injection-probability classifier over the **normalized** user text, run
  only when fast path did not already block (efficiency) and the stage is
  available + enabled. Same block / warn semantics as stage 2; when the
  model or onnxruntime is missing the stage degrades gracefully (skipped
  with a one-time warning).

Later agents add more stages on top:

- Agent 4 — canary token detection
- Agent 6 — pipeline orchestration (block / warn / log-only)

Note on stage interaction: stage 1 already neutralizes raw special tokens
(``<|im_start|>`` → ``⟦fake-im-start⟧``), so fast-path patterns aimed at raw
tokens mainly fire when the corresponding normalization toggles are disabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..config import DeepPathSettings, FastPathSettings, NormalizationSettings
from ..deep_path import DeepPathDetector, DeepPathResult
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
    # Agent 10 (audit): which stage produced the decision and its scores,
    # populated for block decisions (and warn events, via the on_event hook).
    stage: str | None = None
    scores: dict[str, Any] | None = None


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
    """Runs stages over a chat completion payload: normalization, fast path,
    then the deep path (ML) when configured."""

    def __init__(
        self,
        normalization: NormalizationSettings | None = None,
        *,
        fast_path: FastPathSettings | None = None,
        deep_path_detector: DeepPathDetector | None = None,
    ) -> None:
        self.normalization = normalization or NormalizationSettings()
        self.fast_path: FastPathDetector | None = None
        if fast_path is not None and fast_path.enabled:
            self.fast_path = FastPathDetector(
                block_threshold=fast_path.block_threshold,
                warn_threshold=fast_path.warn_threshold,
            )
        # Agent 5 — stage 3: the caller (src.main / tests) builds the
        # detector from settings, so the pipeline stays decoupled from
        # onnxruntime. An unusable model degrades to `available=False` and
        # is skipped with a one-time warning (PLAN.md §3.1).
        self.deep_path = deep_path_detector
        if self.deep_path is not None and not self.deep_path.available:
            logger.warning(
                "Deep path stage disabled: model backend unavailable "
                "(download it with scripts/download_model.py or set "
                "LMPI_DEEP_PATH_ENABLED=false)"
            )

    async def process_request(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> PipelineResult:
        """Inspect a chat completion payload and decide pass / block.

        Stage 1 normalizes every user-message content, rewrites the payload
        with the cleaned text, and logs the findings. Stage 2 runs the fast
        path over the *normalized* user text. Stage 3 runs the deep path ML
        classifier over the same text, only when fast path did not block and
        the stage is available. Depending on the configured stage actions
        the result can block (403), rewrite, or pass through.

        ``request_id`` / ``on_event`` (Agent 10, audit): when ``on_event`` is
        given, every stage-level detection event (findings, warn, block) is
        also delivered as a JSON-safe dict with ``request_id``, ``stage``,
        ``action`` and ``scores`` — used by the structured audit log. The
        existing plain log lines are unchanged.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return PipelineResult()

        def emit_event(event: dict[str, Any]) -> None:
            if on_event is None:
                return
            event = dict(event)
            if request_id is not None:
                event["request_id"] = request_id
            on_event(event)

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
            scores: dict[str, Any] = {
                "findings": [finding.to_dict() for finding in findings]
            }
            if self.normalization.mode == "block":
                emit_event(
                    {"stage": "normalization", "action": "block", "scores": scores}
                )
            else:
                emit_event(
                    {"stage": "normalization", "action": "warn", "scores": scores}
                )

        if findings and self.normalization.mode == "block":
            categories = sorted({finding.category for finding in findings})
            return PipelineResult(
                action="block",
                reason=(
                    "Request blocked by normalization: " + ", ".join(categories)
                ),
                stage="normalization",
                scores=scores,
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
                if detection.action != "allow":
                    self._log_detection(detection, level="warning" if detection.action == "block" else "info")
                    emit_event(
                        {
                            "stage": "fast_path",
                            "action": detection.action,
                            "scores": {
                                key: value
                                for key, value in detection.log_dict().items()
                                if key not in ("stage", "action")
                            },
                        }
                    )
                if detection.action == "block":
                    return PipelineResult(
                        action="block",
                        reason=detection.reason,
                        stage="fast_path",
                        scores={
                            key: value
                            for key, value in detection.log_dict().items()
                            if key not in ("stage", "action")
                        },
                    )

        # ---------------------------------------------------------------
        # Stage 3 — deep path (ML classifier, Agent 5). Skipped when fast
        # path already blocked (a `return` above), when no detector is
        # configured, or when the backend is unavailable (graceful
        # degradation). Same block / warn semantics as stage 2.
        # ---------------------------------------------------------------
        if self.deep_path is not None and self.deep_path.available:
            text = extract_user_text({"messages": new_messages})
            if text.strip():
                detection = self.deep_path.detect(text)
                if detection.action != "allow":
                    level = "warning" if detection.action == "block" else "info"
                    self._log_deep_detection(detection, level=level)
                    emit_event(
                        {
                            "stage": "deep_path",
                            "action": detection.action,
                            "scores": {
                                key: value
                                for key, value in detection.log_dict().items()
                                if key not in ("stage", "action")
                            },
                        }
                    )
                if detection.action == "block":
                    return PipelineResult(
                        action="block",
                        reason=detection.reason,
                        stage="deep_path",
                        scores={
                            key: value
                            for key, value in detection.log_dict().items()
                            if key not in ("stage", "action")
                        },
                    )

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

    # ------------------------------------------------------------------
    # Stage 3 helpers (deep path)
    # ------------------------------------------------------------------

    def _log_deep_detection(
        self, detection: DeepPathResult, level: str
    ) -> None:
        """Emit the structured deep-path detection event (block or warn)."""
        message = json.dumps(
            detection.log_dict(), ensure_ascii=False, sort_keys=True
        )
        if level == "warning":
            logger.warning("detection event: %s", message)
        else:
            logger.info("detection event: %s", message)
