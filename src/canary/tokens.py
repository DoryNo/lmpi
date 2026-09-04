"""Canary token generation and verification.

A canary is a short, naturally-worded audit string embedded into the system
prompt before a request is forwarded upstream::

    [Internal audit token: LMPI-CANARY-ab12cd34]

If the model ever repeats the token in its output (a system prompt leak), the
proxy detects the exact value in the response and redacts or blocks it.

Design (see AGENTS.md, "Agent 4"):

- **Per-request tokens** (default): the identifier is derived as
  ``HMAC-SHA256(secret, random_per_request_salt)`` truncated to 4 bytes.
  Trade-off: every request gets a fresh token, so a leak cannot be correlated
  across requests and a leaked value is useless outside the one response that
  carried it — at the cost of a tiny HMAC per request. A per-session token
  would be marginally cheaper but would make leaked values reusable as
  cross-request fingerprints of the deployment, which we deliberately avoid.
- **Fixed-length ASCII**: the value is always ``LMPI-CANARY-`` + 8 hex chars
  = 20 bytes, so the streaming scanner (``src.canary.manager.CanaryScanState``)
  can size its hold-back buffer deterministically.
- The ``fingerprint`` logged with alerts comes from a *different* slice of the
  HMAC digest than the identifier, so logs never contain the raw canary value.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Union

# Visible token format: LMPI-CANARY- + 8 lowercase hex chars.
CANARY_PREFIX = "LMPI-CANARY-"
CANARY_ID_BYTES = 4
CANARY_ID_CHARS = CANARY_ID_BYTES * 2

# Fixed total length in characters == bytes (pure ASCII), documented
# guarantee used by the streaming hold-back buffer.
CANARY_LENGTH = len(CANARY_PREFIX) + CANARY_ID_CHARS

CANARY_PATTERN = re.compile(r"^LMPI-CANARY-[0-9a-f]{8}$")

CANARY_SENTENCE_TEMPLATE = "[Internal audit token: {value}]"

TokenLike = Union["CanaryToken", str]


@dataclass(frozen=True)
class CanaryToken:
    """One generated canary.

    Attributes:
        value: The exact token string to look for in responses
            (``LMPI-CANARY-<hex8>``). Fixed length, pure ASCII.
        fingerprint: Short log-safe identifier derived from a different part
            of the same HMAC digest — stable, but never equal to ``value``.
    """

    value: str
    fingerprint: str

    def __post_init__(self) -> None:
        if len(self.value) != CANARY_LENGTH or not CANARY_PATTERN.match(self.value):
            raise ValueError(f"invalid canary value: {self.value!r}")

    def sentence(self) -> str:
        """The exact sentence injected into the system prompt."""
        return CANARY_SENTENCE_TEMPLATE.format(value=self.value)


def generate_canary(secret: bytes | str, salt: bytes | str) -> CanaryToken:
    """Derive a canary token from ``HMAC-SHA256(secret, salt)``.

    Deterministic for the same ``(secret, salt)`` pair; callers pass a fresh
    random salt per request (see :meth:`CanaryManager.inject`).
    """
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    if isinstance(salt, str):
        salt = salt.encode("utf-8")
    digest = hmac.new(secret, salt, hashlib.sha256).digest()
    value = CANARY_PREFIX + digest[:CANARY_ID_BYTES].hex()
    # Fingerprint from a disjoint slice: logging it never reveals the id.
    fingerprint = digest[CANARY_ID_BYTES : 3 * CANARY_ID_BYTES].hex()
    return CanaryToken(value=value, fingerprint=fingerprint)


def random_salt(nbytes: int = 16) -> bytes:
    """Fresh random salt for per-request token derivation."""
    return secrets.token_bytes(nbytes)


def ephemeral_secret(nbytes: int = 32) -> bytes:
    """Random secret used when no ``LMPI_CANARY_SECRET`` is configured."""
    return secrets.token_bytes(nbytes)


def verify_text(text: str, token: TokenLike) -> bool:
    """True when ``text`` contains the exact canary ``token`` value.

    Accepts a :class:`CanaryToken` or the raw value string. Substrings such
    as ``LMPI`` or a truncated ``LMPI-CANARY-<hex7>`` do not count.
    """
    value = token.value if isinstance(token, CanaryToken) else token
    return value in text


def is_canary_value(value: str) -> bool:
    """True when ``value`` matches the full fixed canary format."""
    return bool(CANARY_PATTERN.match(value))
