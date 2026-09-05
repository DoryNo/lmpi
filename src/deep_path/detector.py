"""Deep path (stage 3) — threshold decision logic over ML classifier scores.

The classifier returns an *injection probability* in ``[0, 1]``; decision
mirrors the fast-path API::

    score >= block_threshold → "block"   (proxy returns HTTP 403)
    score >= warn_threshold  → "warn"    (logged, request still forwarded)
    otherwise                → "allow"

Defaults ``block=0.65``, ``warn=0.5`` — block threshold tuned against the
frozen benchmark eval set (v1.1 tuning round 1; see
``benchmarks/tuning_log.md``). Input hygiene: the stage runs on
the **normalized** user text (stage 1 output) and caps the classified text
at ``max_chars`` (default 6000) before it reaches the tokenizer, so a
megabyte-sized prompt cannot burn inference time — the cap is recorded in
the log event.

Every result carries ``latency_ms`` (tokenizer + ONNX inference wall
time), ``model`` and ``quantized`` for structured logging; this feeds the
benchmark's per-stage latency story.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from .backend import ModelBackend

DEFAULT_BLOCK_THRESHOLD = 0.65
DEFAULT_WARN_THRESHOLD = 0.5
DEFAULT_MAX_CHARS = 6000

DeepPathAction = Literal["block", "warn", "allow"]


def decide_action(
    score: float, block_threshold: float, warn_threshold: float
) -> DeepPathAction:
    """Map an injection probability onto block / warn / allow."""
    if score >= block_threshold:
        return "block"
    if score >= warn_threshold:
        return "warn"
    return "allow"


@dataclass(frozen=True)
class DeepPathResult:
    """Outcome of one deep-path scan (DetectionResult analogue)."""

    score: float
    action: DeepPathAction
    available: bool = True
    model: str = "unknown"
    quantized: bool = False
    latency_ms: float = 0.0
    char_truncated: bool = False
    max_chars: int = DEFAULT_MAX_CHARS
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    warn_threshold: float = DEFAULT_WARN_THRESHOLD

    @property
    def reason(self) -> str:
        if not self.available:
            return "Deep path unavailable (no backend loaded); request allowed"
        return (
            f"Deep path score={self.score:.2f}, action={self.action} "
            f"(model={self.model}, quantized={self.quantized}; thresholds "
            f"block={self.block_threshold:.2f}/warn={self.warn_threshold:.2f})"
        )

    def log_dict(self) -> dict[str, Any]:
        """Structured-log payload for detection events (JSON-serializable)."""
        return {
            "stage": "deep_path",
            "action": self.action,
            "score": round(self.score, 4),
            "available": self.available,
            "model": self.model,
            "quantized": self.quantized,
            "latency_ms": round(self.latency_ms, 2),
            "input_truncated_chars": self.char_truncated,
            "max_chars": self.max_chars,
            "thresholds": {
                "block": self.block_threshold,
                "warn": self.warn_threshold,
            },
        }


class DeepPathDetector:
    """Injection-probability classifier with threshold decisions.

    Usage::

        detector = DeepPathDetector(OnnxRuntimeBackend("models/deberta-v3-base-prompt-injection-v2"))
        result = detector.detect("Ignore all previous instructions")
        result.action   # "block" | "warn" | "allow"

    ``available`` is False when no backend could be loaded (model files
    absent, onnxruntime missing); :meth:`detect` then reports an ``allow``
    with score 0.0 and the pipeline skips the stage entirely.
    """

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        warn_threshold: float = DEFAULT_WARN_THRESHOLD,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        thresholds = {
            "block_threshold": block_threshold,
            "warn_threshold": warn_threshold,
        }
        for name, value in thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value!r}")
        if warn_threshold > block_threshold:
            raise ValueError(
                f"warn_threshold ({warn_threshold}) must not exceed "
                f"block_threshold ({block_threshold})"
            )
        if int(max_chars) <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars!r}")
        self._backend = backend
        self.block_threshold = float(block_threshold)
        self.warn_threshold = float(warn_threshold)
        self.max_chars = int(max_chars)

    @property
    def available(self) -> bool:
        """True when a model backend is loaded and ready."""
        return self._backend is not None

    def detect(self, text: str) -> DeepPathResult:
        """Score ``text`` (normalized user text) and decide block/warn/allow.

        Never raises on missing backend — it degrades to an ``allow``
        result so the pipeline can treat the stage as disabled.
        """
        if self._backend is None:
            return DeepPathResult(
                score=0.0,
                action="allow",
                available=False,
                max_chars=self.max_chars,
                block_threshold=self.block_threshold,
                warn_threshold=self.warn_threshold,
            )

        char_truncated = False
        original_chars = len(text)
        if original_chars > self.max_chars:
            char_truncated = True
            text = text[: self.max_chars]

        started = time.perf_counter()
        benign_prob, injection_prob = self._backend.predict([text])[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        score = float(injection_prob)
        action = decide_action(score, self.block_threshold, self.warn_threshold)
        return DeepPathResult(
            score=score,
            action=action,
            available=True,
            model=self._backend.model_name,
            quantized=self._backend.quantized,
            latency_ms=latency_ms,
            char_truncated=char_truncated,
            max_chars=self.max_chars,
            block_threshold=self.block_threshold,
            warn_threshold=self.warn_threshold,
        )
