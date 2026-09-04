"""Ingress normalization — stage 1 of the LMPI detection pipeline.

Public API: :func:`normalize` cleans a piece of user content and returns a
:class:`NormalizationResult` (the cleaned text plus structured findings).

Pipeline order — each stage feeds the next:

1. :mod:`.unicode` — NFKC, zero-width, bidi and control-character removal,
   so hidden characters can no longer split keywords or corrupt parsing.
2. :mod:`.encoding` — base64 / hex / ROT13 decode-and-recheck, so encoded
   payloads are surfaced as plain text for the later stages.
3. :mod:`.delimiters` — neutralization of pseudo-system delimiter tokens
   (``<|im_start|>``, ``### System``, ``[INST]`` …) so a user message cannot
   fake role boundaries.
"""

from __future__ import annotations

from .delimiters import BRACKET_TOKENS, neutralize_delimiters
from .encoding import SUSPICIOUS_MARKERS, decode_encodings
from .types import (
    CATEGORY_DELIMITER,
    CATEGORY_ENCODING_BASE64,
    CATEGORY_ENCODING_HEX,
    CATEGORY_ENCODING_ROT13,
    CATEGORY_UNICODE_BIDI,
    CATEGORY_UNICODE_CONTROL,
    CATEGORY_UNICODE_NFKC,
    CATEGORY_UNICODE_ZERO_WIDTH,
    Finding,
    NormalizationResult,
)
from .unicode import ZERO_WIDTH_CHARS, clean_unicode

__all__ = [
    "BRACKET_TOKENS",
    "SUSPICIOUS_MARKERS",
    "ZERO_WIDTH_CHARS",
    "CATEGORY_DELIMITER",
    "CATEGORY_ENCODING_BASE64",
    "CATEGORY_ENCODING_HEX",
    "CATEGORY_ENCODING_ROT13",
    "CATEGORY_UNICODE_BIDI",
    "CATEGORY_UNICODE_CONTROL",
    "CATEGORY_UNICODE_NFKC",
    "CATEGORY_UNICODE_ZERO_WIDTH",
    "Finding",
    "NormalizationResult",
    "clean_unicode",
    "decode_encodings",
    "neutralize_delimiters",
    "normalize",
]


def normalize(
    text: str,
    *,
    unicode_cleaning: bool = True,
    base64: bool = True,
    hex: bool = True,
    rot13: bool = True,
    delimiters: bool = True,
) -> NormalizationResult:
    """Normalize user content through all enabled stages, in order.

    Never raises: any string input returns a valid result, and malformed
    encodings are skipped rather than decoded.
    """
    if not isinstance(text, str):
        return NormalizationResult(cleaned_text="" if text is None else str(text))

    cleaned = text
    findings: list[Finding] = []

    if unicode_cleaning:
        result = clean_unicode(cleaned)
        cleaned = result.cleaned_text
        findings.extend(result.findings)

    if base64 or hex or rot13:
        result = decode_encodings(
            cleaned,
            base64_decoding=base64,
            hex_decoding=hex,
            rot13_decoding=rot13,
        )
        cleaned = result.cleaned_text
        findings.extend(result.findings)

    if delimiters:
        result = neutralize_delimiters(cleaned)
        cleaned = result.cleaned_text
        findings.extend(result.findings)

    return NormalizationResult(cleaned_text=cleaned, findings=findings)
