"""Canary injection into system prompts + response leak scanning.

Two halves, both owned by :class:`CanaryManager`:

- **Request path** — :meth:`CanaryManager.inject` appends a fresh per-request
  canary sentence to the system message (string or multipart content). It runs
  AFTER the detection pipeline rewrite, on the final payload dict, right
  before the body is serialized upstream. With no system message it is a
  no-op by default (transparency — the proxy never adds messages the
  application did not send); ``add_missing_system`` opts in.
- **Response path** — :class:`CanaryScanState` is a request-scoped streaming
  scanner: one instance per proxied response, no global mutable state. It
  scans chunks as they pass through without buffering the stream: at most
  ``CANARY_LENGTH - 1`` bytes are held back (a rolling tail) so a canary
  split across chunk boundaries is still caught; everything else is emitted
  immediately. At end of stream the caller flushes the tail.

Actions on detection (``CanarySettings.action``):

- ``redact`` (default) — the canary is replaced with ``[REDACTED]`` in the
  outgoing body/stream and a structured JSON alert is logged (WARNING).
- ``block`` — non-streaming: the response is replaced with a 502 "leak
  detected" error. Streaming: the stream is terminated with an SSE
  ``event: error`` frame (``lmpi_leak_detected``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import CANARY_ACTIONS, CanarySettings
from .tokens import CANARY_LENGTH, CanaryToken, ephemeral_secret, generate_canary, random_salt

logger = logging.getLogger("lmpi.canary")

REDACTED_TEXT = "[REDACTED]"
REDACTED_BYTES = REDACTED_TEXT.encode("ascii")


class CanaryManager:
    """Owns canary settings; creates per-request injection and scan state.

    The manager itself is stateless across requests (read-only settings +
    an immutable secret), so it is safe to share between concurrent requests.
    """

    def __init__(self, settings: CanarySettings | None = None) -> None:
        self.settings = settings or CanarySettings()
        if self.settings.secret:
            self._secret: bytes = self.settings.secret.encode("utf-8")
        else:
            self._secret = ephemeral_secret()
            logger.warning(
                "canary: LMPI_CANARY_SECRET is not set — generated an ephemeral "
                "secret; tokens are valid for this process only and will not "
                "be reproducible after a restart"
            )

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def action(self) -> str:
        return self.settings.action

    # ------------------------------------------------------------------
    # Request path — injection
    # ------------------------------------------------------------------

    def inject(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], CanaryToken | None]:
        """Append a fresh canary sentence to the system message.

        Returns ``(payload_to_forward, token)``. When nothing was injected
        (disabled, no usable system message and ``add_missing_system`` off,
        or no message list) the *original dict object* is returned with
        ``token=None`` so the proxy keeps forwarding the original bytes
        untouched. The input payload is never mutated.
        """
        if not self.enabled:
            return payload, None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload, None

        token = generate_canary(self._secret, random_salt())
        index = self._find_system_index(messages)
        if index is None:
            if not self.settings.add_missing_system:
                return payload, None
            new_messages: list[Any] = [
                {"role": "system", "content": token.sentence()},
                *messages,
            ]
        else:
            new_messages = list(messages)
            new_messages[index] = self._inject_into_message(messages[index], token)

        new_payload = dict(payload)
        new_payload["messages"] = new_messages
        return new_payload, token

    @staticmethod
    def _find_system_index(messages: list[Any]) -> int | None:
        """Index of the first system message with string or multipart content."""
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, (str, list)):
                return index
        return None

    @staticmethod
    def _inject_into_message(message: dict[str, Any], token: CanaryToken) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": f"{content}\n{token.sentence()}"}
        # Multipart content: append the canary as an extra text part.
        return {
            **message,
            "content": [*content, {"type": "text", "text": token.sentence()}],
        }

    # ------------------------------------------------------------------
    # Response path — scanning
    # ------------------------------------------------------------------

    def new_scan_state(self, token: CanaryToken) -> CanaryScanState:
        """Create the request-scoped scanner for one proxied response."""
        return CanaryScanState(token=token, action=self.action)

    def scan_bytes(
        self, body: bytes, token: CanaryToken
    ) -> tuple[bytes, CanaryScanState]:
        """Scan a complete (non-streaming) response body.

        Returns ``(body_to_send, state)``; in ``redact`` mode the body has
        every canary occurrence replaced with ``[REDACTED]``, in ``block``
        mode the caller checks ``state.leaked`` and discards the body in
        favor of a 502 leak response.
        """
        state = self.new_scan_state(token)
        emitted = state.process(body)
        return emitted + state.flush(), state


@dataclass
class CanaryScanState:
    """Rolling scan state for one proxied response (request-scoped).

    Emits upstream bytes immediately while holding back at most
    ``CANARY_LENGTH - 1`` bytes — the longest suffix that could still grow
    into the canary — so matches split across chunks are caught without
    buffering the stream. ``flush()`` must be called at end of stream to
    release the tail.
    """

    token: CanaryToken
    action: str = "redact"
    leaked: bool = False
    occurrences: int = 0
    _pending: bytes = field(default=b"", repr=False)
    _terminal: bool = field(default=False, repr=False)
    _needle: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        if self.action not in CANARY_ACTIONS:
            raise ValueError(
                f"canary action must be one of {', '.join(CANARY_ACTIONS)}, "
                f"got {self.action!r}"
            )
        self._needle = self.token.value.encode("ascii")

    def process(self, chunk: bytes) -> bytes:
        """Feed one upstream chunk; return the bytes safe to send to the client."""
        if self._terminal:
            return b""
        buf = self._pending + chunk
        out = bytearray()
        while True:
            index = buf.find(self._needle)
            if index == -1:
                break
            self.occurrences += 1
            if not self.leaked:
                self.leaked = True
                self._log_alert()
            if self.action == "block":
                # Hard stop: content before the leak is clean and may be
                # forwarded, but everything from the leak onward is dropped
                # and the caller terminates the stream with an error event.
                self._terminal = True
                self._pending = b""
                return bytes(out + buf[:index])
            out += buf[:index]
            out += REDACTED_BYTES
            buf = buf[index + len(self._needle) :]
        hold = self._holdback_length(buf)
        self._pending = buf[len(buf) - hold :] if hold else b""
        out += buf[: len(buf) - hold]
        return bytes(out)

    def flush(self) -> bytes:
        """Release the held-back tail at end of stream."""
        if self._terminal:
            return b""
        pending, self._pending = self._pending, b""
        return pending

    def _holdback_length(self, buf: bytes) -> int:
        """Longest suffix of ``buf`` that is a proper prefix of the canary.

        Any partial match not ending at the buffer end can never complete
        (the canary is a fixed byte string), so only the suffix needs to be
        held back — at most ``CANARY_LENGTH - 1`` bytes.
        """
        limit = min(len(buf), CANARY_LENGTH - 1)
        for length in range(limit, 0, -1):
            if buf[-length:] == self._needle[:length]:
                return length
        return 0

    def _log_alert(self) -> None:
        event = {
            "stage": "canary",
            "action": self.action,
            "fingerprint": self.token.fingerprint,
            "occurrences": self.occurrences,
        }
        logger.warning(
            "detection event: %s", json.dumps(event, ensure_ascii=False, sort_keys=True)
        )
