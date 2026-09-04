"""Fast path — stage 2 of the LMPI detection pipeline.

Regex/heuristic detection of known jailbreak constructs with weighted
scoring and minimized false positives (see AGENTS.md, "Agent 3").

Public API::

    from src.fast_path import FastPathDetector

    detector = FastPathDetector()               # default thresholds
    result = detector.detect(user_text)         # -> DetectionResult
    result.action    # "block" | "warn" | "allow"
    result.score     # composite noisy-OR score in [0, 1]
    result.matches   # every fired pattern, for transparent logging

Pattern weights are heuristics pending tuning against the frozen benchmark
eval set (Agent 7) — see ``patterns.py`` and the README Fast Path section.
"""

from __future__ import annotations

from .detector import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_WARN_THRESHOLD,
    DetectionResult,
    FastPathAction,
    FastPathDetector,
    PatternMatch,
    combine_weights,
    decide_action,
)
from .patterns import (
    CATEGORIES,
    PATTERNS,
    Pattern,
    WEIGHT_FAKE_ROLE_INJECTION,
    WEIGHT_INSTRUCTION_OVERRIDE,
    WEIGHT_OBFUSCATION_MARKER,
    WEIGHT_ROLEPLAY_JAILBREAK,
    WEIGHT_SYSTEM_PROMPT_EXTRACTION,
    patterns_by_category,
)

__all__ = [
    "CATEGORIES",
    "DEFAULT_BLOCK_THRESHOLD",
    "DEFAULT_WARN_THRESHOLD",
    "DetectionResult",
    "FastPathAction",
    "FastPathDetector",
    "PATTERNS",
    "Pattern",
    "PatternMatch",
    "WEIGHT_FAKE_ROLE_INJECTION",
    "WEIGHT_INSTRUCTION_OVERRIDE",
    "WEIGHT_OBFUSCATION_MARKER",
    "WEIGHT_ROLEPLAY_JAILBREAK",
    "WEIGHT_SYSTEM_PROMPT_EXTRACTION",
    "combine_weights",
    "decide_action",
    "patterns_by_category",
]
