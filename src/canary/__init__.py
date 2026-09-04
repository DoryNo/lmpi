"""Canary token system — Agent 4 of the LMPI detection pipeline.

Detects system prompt leakage: a fresh per-request canary is injected into
the system prompt on the way upstream and the response (streaming or not) is
scanned for the exact token; leaks are redacted or blocked and logged.

Public API::

    from src.canary import CanaryManager

    manager = CanaryManager(settings.canary)
    payload, token = manager.inject(payload)          # request path
    state = manager.new_scan_state(token)             # response path (streaming)
    body, state = manager.scan_bytes(body, token)     # response path (whole body)

See ``tokens.py`` for the token format and ``manager.py`` for the scan state
machine.
"""

from __future__ import annotations

from .manager import REDACTED_BYTES, REDACTED_TEXT, CanaryManager, CanaryScanState
from .tokens import (
    CANARY_ID_BYTES,
    CANARY_ID_CHARS,
    CANARY_LENGTH,
    CANARY_PREFIX,
    CANARY_PATTERN,
    CANARY_SENTENCE_TEMPLATE,
    CanaryToken,
    ephemeral_secret,
    generate_canary,
    is_canary_value,
    random_salt,
    verify_text,
)

__all__ = [
    "CANARY_ID_BYTES",
    "CANARY_ID_CHARS",
    "CANARY_LENGTH",
    "CANARY_PREFIX",
    "CANARY_PATTERN",
    "CANARY_SENTENCE_TEMPLATE",
    "CanaryManager",
    "CanaryScanState",
    "CanaryToken",
    "REDACTED_BYTES",
    "REDACTED_TEXT",
    "ephemeral_secret",
    "generate_canary",
    "is_canary_value",
    "random_salt",
    "verify_text",
]
