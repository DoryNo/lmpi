"""Shared types for the ingress normalization stage.

A *Finding* is one structured observation about the input text: something was
rewritten, removed, or decoded so the downstream detection stages (fast path,
deep path) and the structured logger see exactly what happened. Positions are
spans into the text as it was *when the finding was recorded* — stages run in
sequence (NFKC → zero-width/control removal → encoding decode → delimiter
neutralization), so each stage reports positions against its own input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Categories used by the normalization stages. Exported so log pipelines and
# tests can reference them without string literals drifting apart.
CATEGORY_UNICODE_NFKC = "unicode-nfkc"
CATEGORY_UNICODE_ZERO_WIDTH = "unicode-zero-width"
CATEGORY_UNICODE_CONTROL = "unicode-control"
CATEGORY_UNICODE_BIDI = "unicode-bidi"
CATEGORY_ENCODING_BASE64 = "encoding-base64"
CATEGORY_ENCODING_HEX = "encoding-hex"
CATEGORY_ENCODING_ROT13 = "encoding-rot13"
CATEGORY_DELIMITER = "delimiter-neutralized"


@dataclass(frozen=True)
class Finding:
    """One normalization observation about the input text.

    Attributes:
        category: Machine-readable kind, e.g. ``unicode-zero-width`` or
            ``encoding-base64`` (see the ``CATEGORY_*`` constants).
        description: Human-readable explanation for logs and debug output.
        positions: ``(start, end)`` spans into the stage input text the
            finding refers to. Empty when not applicable (e.g. NFKC, which
            rewrites the whole string).
        preview: Short sanitized preview of decoded/replacement content for
            structured logging (newlines and other control chars escaped).
    """

    category: str
    description: str
    positions: tuple[tuple[int, int], ...] = ()
    preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for structured logging."""
        return {
            "category": self.category,
            "description": self.description,
            "positions": [list(span) for span in self.positions],
            "preview": self.preview,
        }


@dataclass
class NormalizationResult:
    """Outcome of normalizing one piece of user content.

    Attributes:
        cleaned_text: Rewritten text safe to hand to the detection stages and
            the upstream LLM. Equal to the input when nothing was found.
        findings: Every finding recorded while producing ``cleaned_text``.
    """

    cleaned_text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when any transformation was applied (findings recorded)."""
        return bool(self.findings)


def make_preview(text: str, limit: int = 80) -> str:
    """Truncate ``text`` to ``limit`` chars with control chars escaped."""
    escaped = "".join(
        f"\\u{ord(ch):04x}" if (ord(ch) < 0x20 or ord(ch) == 0x7F) else ch
        for ch in text
    )
    if len(escaped) <= limit:
        return escaped
    return escaped[: limit - 1] + "…"
