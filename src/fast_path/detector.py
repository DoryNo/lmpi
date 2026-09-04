"""Weighted scoring engine for fast-path detection.

Composite score is a **noisy-OR** over the effective weights of the matched
patterns::

    score = 1 - Π(1 - w_i)      (each w_i clamped to [0, 1]; result in [0, 1])

Multiple weak signals stack multiplicatively: three 0.5-weight obfuscation
hits combine to 0.875, so evasion techniques used together still block.

Decision (defaults ``block=0.75``, ``warn=0.4`` — configurable via
``LMPI_FAST_PATH_BLOCK_THRESHOLD`` / ``LMPI_FAST_PATH_WARN_THRESHOLD`` or
``config.yaml``)::

    score >= block → "block"   (proxy returns HTTP 403)
    score >= warn  → "warn"    (logged, request still forwarded)
    otherwise      → "allow"

All weights are **heuristics** pending tuning against the frozen benchmark
eval set (Agent 7) — see PLAN.md §4.1 and README's Fast Path section.

Transparency: every fired pattern (including quoted *mentions* that were
demoted to zero weight) is carried on the result, so logs can always answer
"what triggered this decision".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .patterns import PATTERNS, Pattern

DEFAULT_BLOCK_THRESHOLD = 0.75
DEFAULT_WARN_THRESHOLD = 0.4

# A verbatim pattern hit inside quotation marks is treated as a *mention*
# (e.g. security coursework quoting "ignore all previous instructions")
# rather than an instruction, and contributes
# MENTION_DEMOTION_FACTOR * base_weight (0.0 = fully discounted) to the
# score. The match is still recorded for transparency.
#
# Known trade-off: an attacker who wraps their instruction in quotes evades
# the fast path. That is accepted deliberately — false positives are the v1
# priority, and the deep-path ML stage (Agent 5) is expected to cover
# mention-shaped attacks.
MENTION_DEMOTION_FACTOR = 0.0

_SNIPPET_LIMIT = 120

_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("“", "”"),
    ("«", "»"),
    ("‘", "’"),
    ("`", "`"),
    ("'", "'"),
)


def combine_weights(weights: Iterable[float]) -> float:
    """Noisy-OR combination: ``1 - Π(1 - w_i)``, clamped to ``[0, 1]``."""
    combined = 0.0
    for weight in weights:
        w = min(max(float(weight), 0.0), 1.0)
        combined = 1.0 - (1.0 - combined) * (1.0 - w)
    return min(max(combined, 0.0), 1.0)


FastPathAction = Literal["block", "warn", "allow"]


def decide_action(
    score: float, block_threshold: float, warn_threshold: float
) -> FastPathAction:
    """Map a composite score onto block / warn / allow."""
    if score >= block_threshold:
        return "block"
    if score >= warn_threshold:
        return "warn"
    return "allow"


@dataclass(frozen=True)
class PatternMatch:
    """One fired pattern hit, with its effective (post-demotion) weight."""

    pattern_id: str
    category: str
    weight: float
    base_weight: float
    matched_text: str
    span: tuple[int, int]
    demoted: bool = False


@dataclass(frozen=True)
class DetectionResult:
    """Full-transparency outcome of a fast-path scan."""

    score: float
    action: FastPathAction
    matches: tuple[PatternMatch, ...] = ()
    categories: frozenset[str] = frozenset()
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    warn_threshold: float = DEFAULT_WARN_THRESHOLD

    @property
    def pattern_ids(self) -> tuple[str, ...]:
        """Stable ids of every fired pattern, in registry order."""
        return tuple(match.pattern_id for match in self.matches)

    @property
    def reason(self) -> str:
        categories = ", ".join(sorted(self.categories)) or "none"
        return (
            f"Fast path score={self.score:.2f}, action={self.action} "
            f"(thresholds block={self.block_threshold:.2f}/"
            f"warn={self.warn_threshold:.2f}; categories: {categories})"
        )

    def log_dict(self) -> dict[str, Any]:
        """Structured-log payload for detection events (JSON-serializable)."""
        return {
            "stage": "fast_path",
            "action": self.action,
            "score": round(self.score, 4),
            "categories": sorted(self.categories),
            "patterns": [
                {
                    "id": match.pattern_id,
                    "category": match.category,
                    "weight": round(match.weight, 3),
                    "demoted": match.demoted,
                }
                for match in self.matches
            ],
            "thresholds": {
                "block": self.block_threshold,
                "warn": self.warn_threshold,
            },
        }


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """Return character spans covered by quotation marks.

    Straight quotes pair sequentially (open/close alternating); paired marks
    ("" « » '' ``) pair in order of appearance. This is a heuristic — text
    with stray apostrophes can mis-pair, which is acceptable for a
    demotion-only mechanism.
    """
    spans: list[tuple[int, int]] = []
    for open_quote, close_quote in _QUOTE_PAIRS:
        cursor = 0
        while True:
            start = text.find(open_quote, cursor)
            if start == -1:
                break
            end = text.find(close_quote, start + len(open_quote))
            if end == -1:
                break
            spans.append((start, end + len(close_quote)))
            cursor = end + len(close_quote)
    return spans


def _span_is_quoted(span: tuple[int, int], quoted: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(quote_start <= start and end <= quote_end for quote_start, quote_end in quoted)


def _snippet(text: str, span: tuple[int, int]) -> str:
    fragment = text[span[0] : span[1]]
    if len(fragment) > _SNIPPET_LIMIT:
        fragment = fragment[: _SNIPPET_LIMIT - 1] + "…"
    return fragment


class FastPathDetector:
    """Regex/heuristic jailbreak detector with noisy-OR scoring.

    Usage::

        detector = FastPathDetector()          # default thresholds
        result = detector.detect("Ignore all previous instructions")
        result.action   # "block"
    """

    def __init__(
        self,
        *,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        warn_threshold: float = DEFAULT_WARN_THRESHOLD,
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
        self.block_threshold = float(block_threshold)
        self.warn_threshold = float(warn_threshold)

    def detect(self, text: str) -> DetectionResult:
        """Scan ``text`` and return the composite score + decision."""
        text = text or ""
        quoted = _quoted_spans(text)

        # Collect raw hits: one contribution per pattern (duplicates of the
        # same pattern id must not inflate the score).
        raw: list[tuple[Pattern, re.Match[str]]] = []
        for pattern in PATTERNS:
            for match in pattern.regex.finditer(text):
                if pattern.predicate is not None and not pattern.predicate(match):
                    continue
                raw.append((pattern, match))
                break

        # Structural gating: requirements are evaluated against the tags of
        # all raw hits (before gating), which makes mutual requirements
        # (persona ⇄ lift) work.
        raw_tags = {tag for pattern, _ in raw for tag in pattern.tags}

        matches: list[PatternMatch] = []
        for pattern, match in raw:
            if pattern.requires and not all(
                requirement in raw_tags or requirement in pattern.tags
                for requirement in pattern.requires
            ):
                continue
            demoted = _span_is_quoted(match.span(), quoted)
            weight = (
                pattern.weight * MENTION_DEMOTION_FACTOR if demoted else pattern.weight
            )
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    category=pattern.category,
                    weight=weight,
                    base_weight=pattern.weight,
                    matched_text=_snippet(text, match.span()),
                    span=(match.start(), match.end()),
                    demoted=demoted,
                )
            )

        contributing = [match for match in matches if match.weight > 0]
        score = combine_weights(match.weight for match in contributing)
        categories = frozenset(match.category for match in contributing)
        action = decide_action(score, self.block_threshold, self.warn_threshold)
        return DetectionResult(
            score=score,
            action=action,
            matches=tuple(matches),
            categories=categories,
            block_threshold=self.block_threshold,
            warn_threshold=self.warn_threshold,
        )
