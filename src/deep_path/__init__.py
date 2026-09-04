"""Deep path — stage 3 of the LMPI detection pipeline.

Quantized-ish ONNX ML classifier (DeBERTa prompt-injection, inference via
``onnxruntime``) for prompt-injection signals the regex fast path misses
(see AGENTS.md, "Agent 5").

Public API::

    from src.deep_path import DeepPathDetector, OnnxRuntimeBackend

    backend = OnnxRuntimeBackend("models/deberta-v3-base-prompt-injection-v2")
    detector = DeepPathDetector(backend)          # default thresholds
    result = detector.detect(user_text)           # -> DeepPathResult
    result.action    # "block" | "warn" | "allow"
    result.score     # injection probability in [0, 1]
    result.latency_ms  # inference wall time, feeds the benchmark

The model is a *pretrained* classifier, NOT fine-tuned on LMPI data —
honesty note in README/PLAN.md §3.3. Enable it with
``python scripts/download_model.py`` then ``LMPI_DEEP_PATH_ENABLED=true``.
When the model or onnxruntime is missing, the stage degrades gracefully:
``DeepPathDetector(available=False)`` and the pipeline skips it with a
one-time warning.
"""

from __future__ import annotations

from .backend import MODEL_FILENAME_PLAIN, MODEL_FILENAME_QUANTIZED, ModelBackend, OnnxRuntimeBackend, StubBackend, softmax_rows
from .detector import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_MAX_CHARS,
    DEFAULT_WARN_THRESHOLD,
    DeepPathAction,
    DeepPathDetector,
    DeepPathResult,
    decide_action,
)

__all__ = [
    "DEFAULT_BLOCK_THRESHOLD",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_WARN_THRESHOLD",
    "MODEL_FILENAME_PLAIN",
    "MODEL_FILENAME_QUANTIZED",
    "DeepPathAction",
    "DeepPathDetector",
    "DeepPathResult",
    "ModelBackend",
    "OnnxRuntimeBackend",
    "StubBackend",
    "decide_action",
    "softmax_rows",
]
