"""Encoding decode-and-recheck — stage 2 of ingress normalization.

Attackers hide instructions in base64, hex or ROT13 so keyword-based filters
see only noise. This stage finds encoded blobs, decodes them, and — when the
decoded content looks like real text — rewrites the message with the decoded
content so the downstream stages (delimiter neutralization, fast path, deep
path) can inspect what the LLM would actually end up reasoning about.

Design decisions:

- **base64 / hex**: a run of the respective alphabet (min length 12 / 20)
  that decodes to strict UTF-8 with mostly-printable content is replaced by
  its decoded text; malformed blobs are skipped silently (never crash).
- **ROT13**: decoding every word would be noisy, so a run is rewritten only
  when its decoded text contains a suspicious marker ("ignore", "system",
  "jailbreak", ...). See :data:`SUSPICIOUS_MARKERS`.
- **Nested encodings**: the rewrite loop runs again over the result, up to
  ``MAX_DECODE_DEPTH`` layers (base64(base64("..."))).
- **Decoded content is the dangerous surface**: after decoding a blob, the
  delimiter neutralizer also runs over the decoded text, so an inlined
  ``System: ...`` payload cannot rely on sitting mid-line to escape the
  line-start anchor of the delimiter stage.
- Hex blobs must be continuous (``68656c6c6f``); space-separated hex bytes
  are deliberately not decoded (too many benign false positives, e.g. dates).
"""

from __future__ import annotations

import base64
import codecs
import re
from binascii import Error as BinasciiError

from .types import (
    CATEGORY_ENCODING_BASE64,
    CATEGORY_ENCODING_HEX,
    CATEGORY_ENCODING_ROT13,
    Finding,
    NormalizationResult,
    make_preview,
)
from .delimiters import neutralize_delimiters as _neutralize_delimiters

# A blob must be bounded by non-alphabet chars so we never decode the middle
# of a longer word/identifier. Alphabet covers standard and URL-safe base64;
# the "=" padding may only trail the run. The 12-char minimum keeps short
# tokens (JWT headers aside) out while still catching "aGVsbG8gd29ybGQ="
# (15 alphabet chars + padding).
_BLOB_RE = re.compile(
    r"(?<![A-Za-z0-9+/_-])"          # not preceded by an alphabet char
    r"([A-Za-z0-9+/_-]{12,}={0,2})"  # the candidate blob itself
    r"(?![A-Za-z0-9+/_-])"           # not followed by an alphabet char
)

# ROT13 candidates: letter runs. Only rewritten when the decode is suspicious.
_ROT13_RE = re.compile(r"[A-Za-z]{6,}")

_HEX_RE = re.compile(r"[0-9a-fA-F]+")

# Markers that make a ROT13 decode worth rewriting. Kept at >= 5 chars to
# avoid flagging benign short words.
SUSPICIOUS_MARKERS: tuple[str, ...] = (
    "ignore",
    "disregard",
    "forget",
    "override",
    "system",
    "assistant",
    "developer mode",
    "instructions",
    "jailbreak",
    "prompt",
    "reveal",
    "pretend",
    "act as",
    "new rules",
    "no rules",
    "no restrictions",
    "do anything",
    "unfiltered",
    "unrestricted",
)

MIN_BLOB_LENGTH = 12
MIN_HEX_LENGTH = 20
MIN_ROT13_LENGTH = 6  # must match _ROT13_RE
MAX_DECODE_DEPTH = 2
# Fraction of printable (non-whitespace) chars required in decoded text.
MIN_PRINTABLE_RATIO = 0.85

_URLSAFE_TRANS = str.maketrans("-_", "+/")


def contains_suspicious_marker(text: str) -> bool:
    """True when ``text`` (case-insensitive) contains a suspicious marker."""
    lowered = text.lower()
    return any(marker in lowered for marker in SUSPICIOUS_MARKERS)


def _as_decodable_text(data: bytes) -> str | None:
    """Return ``data`` as strict UTF-8 text worth rewriting, else ``None``.

    The recheck: garbage decodes (binary payloads, accidental collisions) are
    rejected so benign long words / hashes / numbers are left untouched.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return None
    if not any(ch.isalpha() for ch in non_space):
        return None
    printable = sum(1 for ch in non_space if ch.isprintable())
    if printable / len(non_space) < MIN_PRINTABLE_RATIO:
        return None
    return text


def _try_base64(blob: str) -> str | None:
    """Decode a base64 candidate (standard and URL-safe); None if invalid."""
    core = blob.rstrip("=")
    if not core or "=" in core:
        return None
    pad = (-len(core)) % 4
    if pad == 3:  # length % 4 == 1 is never valid base64
        return None
    padded = core + "=" * pad
    try:
        data = base64.b64decode(padded, validate=True)
    except (BinasciiError, ValueError):
        try:
            data = base64.b64decode(
                padded.translate(_URLSAFE_TRANS), validate=True
            )
        except (BinasciiError, ValueError):
            return None
    return _as_decodable_text(data)


def _try_hex(blob: str) -> str | None:
    """Decode an even-length hex run; None if invalid or garbage."""
    if len(blob) < MIN_HEX_LENGTH or len(blob) % 2 != 0:
        return None
    if not _HEX_RE.fullmatch(blob):
        return None
    try:
        data = bytes.fromhex(blob)
    except ValueError:
        return None
    return _as_decodable_text(data)


def _try_rot13(word: str) -> str | None:
    """ROT13-decode ``word``; None unless the decode hits a suspicious marker."""
    decoded = codecs.decode(word, "rot_13")
    if decoded == word:
        return None
    if contains_suspicious_marker(decoded):
        return decoded
    return None


def _decode_blob(blob: str) -> tuple[str | None, str | None]:
    """Return ``(decoded_text, encoding_name)`` for a base64/hex candidate."""
    hex_text = _try_hex(blob)
    if hex_text is not None:
        return hex_text, "hex"
    b64_text = _try_base64(blob)
    if b64_text is not None:
        return b64_text, "base64"
    return None, None


def decode_encodings(
    text: str,
    *,
    base64_decoding: bool = True,
    hex_decoding: bool = True,
    rot13_decoding: bool = True,
) -> NormalizationResult:
    """Find and decode base64 / hex / ROT13 blobs in ``text``.

    Never raises on any input. Returns a :class:`NormalizationResult` whose
    ``cleaned_text`` has decoded blobs inlined; each rewrite is recorded as a
    finding with a sanitized preview of the decoded content.
    """
    findings: list[Finding] = []
    current = text
    for _ in range(MAX_DECODE_DEPTH):
        result = _decode_once(
            current,
            base64_decoding=base64_decoding,
            hex_decoding=hex_decoding,
            rot13_decoding=rot13_decoding,
        )
        if result is None:
            break
        new_text, pass_findings = result
        findings.extend(pass_findings)
        current = new_text
    return NormalizationResult(cleaned_text=current, findings=findings)


def _decode_once(
    text: str,
    *,
    base64_decoding: bool,
    hex_decoding: bool,
    rot13_decoding: bool,
) -> tuple[str, list[Finding]] | None:
    """One decode pass; ``None`` when nothing in ``text`` decodes."""
    # Collect blob spans (base64/hex candidates).
    blob_spans: list[tuple[int, int, str]] = []
    if base64_decoding or hex_decoding:
        for match in _BLOB_RE.finditer(text):
            blob = match.group(1)
            start, end = match.span(1)
            blob_spans.append((start, end, blob))

    # Collect ROT13 candidates from the original text. A candidate that ends
    # up inside a *successfully decoded* span is skipped below (it is b64
    # content, not ROT13); merely-candidate blobs must not block ROT13 —
    # many 12+ letter words look like base64 but decode to garbage.
    rot13_spans: list[tuple[int, int, str]] = []
    if rot13_decoding:
        for match in _ROT13_RE.finditer(text):
            rot13_spans.append((match.start(), match.end(), match.group()))

    if not blob_spans and not rot13_spans:
        return None

    rewrites: dict[tuple[int, int], str] = {}
    pass_findings: list[Finding] = []
    decoded_spans: list[tuple[int, int]] = []

    for start, end, blob in blob_spans:
        if not (base64_decoding or hex_decoding):
            continue
        decoded, encoding_name = _decode_blob(blob)
        if decoded is None or encoding_name is None:
            continue
        if decoded == blob:  # paranoia: never replace a span with itself
            continue
        if not hex_decoding and encoding_name == "hex":
            continue
        if not base64_decoding and encoding_name == "base64":
            continue
        # Decoded content is treated as an injected message: neutralize
        # pseudo-system delimiters inside it before inlining (it sits
        # mid-line, where the delimiter stage's line-start anchor can't
        # reach).
        decoded_result = _neutralize_delimiters(decoded)
        inlined = decoded_result.cleaned_text
        rewrites[(start, end)] = inlined
        pass_findings.append(
            Finding(
                category=(
                    CATEGORY_ENCODING_HEX
                    if encoding_name == "hex"
                    else CATEGORY_ENCODING_BASE64
                ),
                description=f"Decoded {encoding_name} blob and inlined its content",
                positions=((start, end),),
                preview=make_preview(inlined),
            )
        )
        pass_findings.extend(decoded_result.findings)
        decoded_spans.append((start, end))

    for start, end, word in rot13_spans:
        if not rot13_decoding:
            continue
        if any(bs <= start and end <= be for bs, be in decoded_spans):
            continue
        decoded = _try_rot13(word)
        if decoded is None or decoded == word:
            continue
        rewrites[(start, end)] = decoded
        pass_findings.append(
            Finding(
                category=CATEGORY_ENCODING_ROT13,
                description="Decoded ROT13 run containing a suspicious marker",
                positions=((start, end),),
                preview=make_preview(decoded),
            )
        )

    if not pass_findings:
        return None

    # Rebuild the string with decoded spans inlined, sorted by position.
    parts: list[str] = []
    cursor = 0
    for (start, end), replacement in sorted(rewrites.items()):
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), pass_findings
