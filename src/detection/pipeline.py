"""Detection pipeline hook used by the proxy.

Current stages (see PLAN.md §2.3 and AGENTS.md):

1. **Ingress normalization** (Agent 2) — applied when a ``normalize()``
   function exists in ``src/normalization/``. That module is being built in
   parallel on a separate branch; the hook below (``_normalize_text``) is the
   marked insertion point and degrades gracefully to raw text until it lands.
2. **Fast path** (Agent 3) — regex/heuristic jailbreak detection with
   weighted scoring (``src/fast_path/``).

Later stages (canary — Agent 4, deep path ML — Agent 5, decision
orchestration — Agent 6) plug in behind the same pattern: each adds its scan
and the stage mapping stays block (403) / warn (log-only pass) / allow.

The public surface used by the proxy — ``DetectionPipeline.process_request``
returning a :class:`PipelineResult` — is unchanged.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..config import Settings
from ..fast_path import DetectionResult, FastPathDetector

Action = Literal["pass", "block"]

logger = logging.getLogger("lmpi.detection")


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


# ---------------------------------------------------------------------------
# INSERTION POINT — ingress normalization (Agent 2, src/normalization/).
# That module is being built in parallel on a separate branch. Until it is
# merged, requests are scanned on raw text; once ``normalize()`` (or
# ``normalize_text()``) appears in ``src/normalization/``, it is picked up
# automatically with no changes here. A normalization failure never blocks
# a request: we log and fall back to raw text.
# ---------------------------------------------------------------------------

# Sentinel states: False = not loaded yet, None = unavailable, else the callable.
_normalizer: Callable[[str], str] | None | False = False


def _get_normalizer() -> Callable[[str], str] | None:
    global _normalizer
    if _normalizer is False:
        function: Callable[[str], str] | None = None
        try:
            module = importlib.import_module("src.normalization")
            function = getattr(module, "normalize", None) or getattr(
                module, "normalize_text", None
            )
        except ImportError:
            function = None
        _normalizer = function if callable(function) else None
    return _normalizer


def _normalize_text(text: str) -> str:
    normalize = _get_normalizer()
    if normalize is None:
        return text
    try:
        return normalize(text)
    except Exception as exc:  # noqa: BLE001 — any failure must not break proxying
        logger.warning(
            "Ingress normalization failed (%s); fast path runs on raw text",
            type(exc).__name__,
        )
        return text


class DetectionPipeline:
    """Runs request payloads through the detection stages.

    With default settings the fast path is active: user text is scanned and
    a composite score >= the block threshold yields a 403, a score >= the
    warn threshold is logged but forwarded, anything below passes silently.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.fast_path: FastPathDetector | None = None
        if self.settings.fast_path_enabled:
            self.fast_path = FastPathDetector(
                block_threshold=self.settings.fast_path_block_threshold,
                warn_threshold=self.settings.fast_path_warn_threshold,
            )

    async def process_request(self, payload: dict[str, Any]) -> PipelineResult:
        """Inspect a chat completion payload and decide pass / block."""
        if self.fast_path is None:  # fast path disabled in settings
            return PipelineResult()

        text = extract_user_text(payload)
        if not text.strip():
            return PipelineResult()

        result = self.fast_path.detect(_normalize_text(text))
        return self._apply(result)

    def _apply(self, result: DetectionResult) -> PipelineResult:
        if result.action == "block":
            logger.warning(
                "LMPI detection event: %s",
                json.dumps(result.log_dict(), ensure_ascii=False, sort_keys=True),
            )
            return PipelineResult(action="block", reason=result.reason)
        if result.action == "warn":
            logger.info(
                "LMPI detection event: %s",
                json.dumps(result.log_dict(), ensure_ascii=False, sort_keys=True),
            )
        return PipelineResult()
