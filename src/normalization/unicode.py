"""Unicode normalization — stage 1 of ingress normalization.

Cleans the character-level tricks used to smuggle instructions past detectors
and LLMs alike:

1. **NFKC** — collapse compatibility forms (fullwidth ``ＩＧＮＯＲＥ``, ligature
   ``ﬁ`` inside keywords, circled digits, NBSP, …) so pattern matchers see the
   canonical form an attacker tried to disguise.
2. **Zero-width chars** — strip U+200B/U+200C/U+200D, word joiners, soft
   hyphens, invisible math operators and friends (``i\\u200bgnore``).
3. **Bidi controls** — strip U+202A-U+202E / U+2066-U+2069 (Trojan-Source
   style visual reordering).
4. **Control chars** — strip everything in categories ``Cc``/``Cf`` except
   ``\\n`` and ``\\t``.

Every removal is recorded as a :class:`~src.normalization.types.Finding` with
spans into the text as it stood at that point (post-NFKC).
"""

from __future__ import annotations

import unicodedata

from .types import (
    CATEGORY_UNICODE_BIDI,
    CATEGORY_UNICODE_CONTROL,
    CATEGORY_UNICODE_NFKC,
    CATEGORY_UNICODE_ZERO_WIDTH,
    Finding,
    NormalizationResult,
)

# Chars kept by the control-char pass. Everything else in Cc/Cf goes.
_KEEP_CONTROL = {"\n", "\t"}

# Explicit zero-width / invisible separators. Notable members: the classic
# U+200B/U+200C/U+200D family, word joiner U+2060, BOM U+FEFF, soft hyphen
# U+00AD (invisible and frequently used to break keyword matching), the
# invisible math operators U+2061-U+2064, and the deprecated zero-width
# U+180E (Mongolian vowel separator, zero-width in older Unicode versions).
ZERO_WIDTH_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064"
    "\ufeff\u00ad\u180e"
)

# Bidirectional formatting controls (Trojan Source): left-to-right/right-to-
# left embedding overrides and isolates. All invisible, all dangerous.
BIDI_CHARS = frozenset(
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)

# Unicode general categories treated as "control-like" for removal.
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf"})


def _spans_for(predicate, text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Remove chars where ``predicate(ch)`` is true; return (cleaned, spans)."""
    if not any(predicate(ch) for ch in text):
        return text, ()
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, ch in enumerate(text):
        if predicate(ch):
            if run_start is None:
                run_start = index
        else:
            if run_start is not None:
                spans.append((run_start, index))
                run_start = None
            parts.append(ch)
    if run_start is not None:
        spans.append((run_start, len(text)))
    return "".join(parts), tuple(spans)


def _is_control_char(ch: str) -> bool:
    if ch in _KEEP_CONTROL:
        return False
    return unicodedata.category(ch) in _CONTROL_CATEGORIES


def clean_unicode(text: str) -> NormalizationResult:
    """Apply NFKC, then strip zero-width, bidi and control characters.

    Never raises: any input string is accepted and returns a valid result.
    """
    findings: list[Finding] = []
    current = text

    # 1. NFKC — collapses compatibility forms an attacker uses to hide
    #    keywords (fullwidth letters, ligatures, NBSP, keycaps, ...).
    nfkc = unicodedata.normalize("NFKC", current)
    if nfkc != current:
        findings.append(
            Finding(
                category=CATEGORY_UNICODE_NFKC,
                description="NFKC normalization changed the text",
            )
        )
        current = nfkc

    # 2. Zero-width / invisible separators.
    stripped, zero_spans = _spans_for(ZERO_WIDTH_CHARS.__contains__, current)
    if zero_spans:
        findings.append(
            Finding(
                category=CATEGORY_UNICODE_ZERO_WIDTH,
                description=f"Removed {len(zero_spans)} zero-width character run(s)",
                positions=zero_spans,
            )
        )
        current = stripped

    # 3. Bidirectional controls.
    stripped, bidi_spans = _spans_for(BIDI_CHARS.__contains__, current)
    if bidi_spans:
        findings.append(
            Finding(
                category=CATEGORY_UNICODE_BIDI,
                description=f"Removed {len(bidi_spans)} bidi control char run(s)",
                positions=bidi_spans,
            )
        )
        current = stripped

    # 4. Remaining control characters (Cc/Cf), keeping \n and \t.
    stripped, ctrl_spans = _spans_for(_is_control_char, current)
    if ctrl_spans:
        findings.append(
            Finding(
                category=CATEGORY_UNICODE_CONTROL,
                description=f"Removed {len(ctrl_spans)} control char run(s)",
                positions=ctrl_spans,
            )
        )
        current = stripped

    return NormalizationResult(cleaned_text=current, findings=findings)
