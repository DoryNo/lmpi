"""CanaryManager injection + CanaryScanState streaming scanner tests."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.canary import CanaryManager, CanaryScanState, generate_canary
from src.config import CanarySettings

TOKEN = generate_canary(b"scan-secret", b"scan-salt")
NEEDLE = TOKEN.value.encode("ascii")


def make_manager(**overrides: Any) -> CanaryManager:
    return CanaryManager(CanarySettings(secret="test-secret", **overrides))


def scan_chunks(
    chunks: list[bytes], token: Any = TOKEN, action: str = "redact"
) -> tuple[bytes, CanaryScanState]:
    """Feed chunks through a fresh scanner; return (client bytes, state)."""
    state = CanaryScanState(token=token, action=action)
    emitted = b"".join(state.process(chunk) for chunk in chunks)
    return emitted + state.flush(), state


def system_payload(content: Any) -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": "hi"},
        ],
    }


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


class TestInject:
    def test_appends_to_string_system_message(self) -> None:
        manager = make_manager()
        payload = system_payload("You are helpful.")
        new_payload, token = manager.inject(payload)

        assert token is not None
        system_content = new_payload["messages"][0]["content"]
        assert system_content == f"You are helpful.\n{token.sentence()}"
        assert new_payload["messages"][1] == {"role": "user", "content": "hi"}
        # Input payload never mutated.
        assert payload["messages"][0]["content"] == "You are helpful."

    def test_multipart_system_message_gets_extra_text_part(self) -> None:
        manager = make_manager()
        payload = system_payload([{"type": "text", "text": "You are helpful."}])
        new_payload, token = manager.inject(payload)

        assert token is not None
        parts = new_payload["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "You are helpful."}
        assert parts[-1] == {"type": "text", "text": token.sentence()}
        assert len(parts) == 2
        assert new_payload["messages"][1] == {"role": "user", "content": "hi"}

    def test_no_system_message_is_noop_by_default(self) -> None:
        manager = make_manager()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        new_payload, token = manager.inject(payload)

        assert token is None
        assert new_payload is payload  # byte-for-byte transparency preserved

    def test_add_missing_system_prepends_system_message(self) -> None:
        manager = make_manager(add_missing_system=True)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        new_payload, token = manager.inject(payload)

        assert token is not None
        assert new_payload["messages"][0] == {
            "role": "system",
            "content": token.sentence(),
        }
        assert new_payload["messages"][1] == {"role": "user", "content": "hi"}

    def test_disabled_manager_is_noop(self) -> None:
        manager = make_manager(enabled=False)
        payload = system_payload("You are helpful.")
        new_payload, token = manager.inject(payload)

        assert token is None
        assert new_payload is payload

    def test_missing_or_non_list_messages_is_noop(self) -> None:
        manager = make_manager()
        for payload in ({}, {"messages": "not-a-list"}, {"messages": None}):
            new_payload, token = manager.inject(payload)
            assert token is None

    def test_unusable_system_content_treated_as_missing(self) -> None:
        manager = make_manager()
        payload = system_payload(None)
        new_payload, token = manager.inject(payload)
        assert token is None

    def test_unique_token_per_request(self) -> None:
        manager = make_manager()
        payload = system_payload("You are helpful.")
        _, first = manager.inject(payload)
        _, second = manager.inject(payload)
        assert first is not None and second is not None
        assert first.value != second.value
        assert first.fingerprint != second.fingerprint

    def test_ephemeral_secret_warning_logged_once_per_manager(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="lmpi.canary"):
            manager = CanaryManager(CanarySettings(secret=None))
        warnings = [r for r in caplog.records if r.name == "lmpi.canary"]
        assert len(warnings) == 1
        assert "LMPI_CANARY_SECRET" in warnings[0].getMessage()
        # Still functional with the ephemeral secret.
        _, token = manager.inject(system_payload("sys"))
        assert token is not None

    def test_configured_secret_produces_deterministic_tokens(self) -> None:
        manager_a = make_manager()
        manager_b = make_manager()
        # Same secret but per-request salt → unique values; determinism itself
        # is covered at the token level (test_canary_tokens.py).
        _, token_a = manager_a.inject(system_payload("sys"))
        _, token_b = manager_b.inject(system_payload("sys"))
        assert token_a is not None and token_b is not None
        assert token_a.value != token_b.value


# ---------------------------------------------------------------------------
# CanaryScanState — streaming scanner
# ---------------------------------------------------------------------------


class TestScanStateRedact:
    def test_canary_fully_inside_one_chunk(self) -> None:
        emitted, state = scan_chunks([b"before " + NEEDLE + b" after"])
        assert emitted == b"before [REDACTED] after"
        assert state.leaked is True
        assert state.occurrences == 1

    def test_split_across_two_chunks(self) -> None:
        half = len(NEEDLE) // 2
        emitted, state = scan_chunks([b"intro ", NEEDLE[:half], NEEDLE[half:] + b" outro"])
        assert emitted == b"intro [REDACTED] outro"
        assert NEEDLE not in emitted
        assert state.occurrences == 1

    def test_split_across_three_chunks(self) -> None:
        third = len(NEEDLE) // 3  # 6 + 6 + 8 == 20
        chunks = [
            b"a",
            NEEDLE[:third],
            NEEDLE[third : 2 * third],
            NEEDLE[2 * third :],
            b"b",
        ]
        emitted, state = scan_chunks(chunks)
        assert emitted == b"a[REDACTED]b"
        assert state.occurrences == 1

    def test_exact_tail_boundary(self) -> None:
        # Canary ends exactly at the chunk end: still detected, nothing held.
        state = CanaryScanState(token=TOKEN)
        out = state.process(b"pre " + NEEDLE)
        tail = state.flush()
        assert out + tail == b"pre [REDACTED]"
        assert state.occurrences == 1
        assert state.flush() == b""

    def test_multiple_occurrences_in_one_chunk(self) -> None:
        emitted, state = scan_chunks([b"x " + NEEDLE + b" mid " + NEEDLE + b" y"])
        assert emitted == b"x [REDACTED] mid [REDACTED] y"
        assert state.occurrences == 2

    def test_multiple_occurrences_across_chunks(self) -> None:
        emitted, state = scan_chunks([NEEDLE, b" - ", NEEDLE])
        assert emitted == b"[REDACTED] - [REDACTED]"
        assert state.occurrences == 2

    def test_no_false_positive_on_benign_lmpi_text(self) -> None:
        samples = [
            b"LMPI is great",
            b"LMPI-CANARY- is only a prefix",
            b"LMPI-CANARY-1234567",  # 7 hex chars — not a valid canary
            b"LMPI-CANARY-12345678",  # valid format, different token
            b"lmpi-canary-" + NEEDLE[len(b"LMPI-CANARY-") :],  # lowercase prefix
        ]
        for sample in samples:
            emitted, state = scan_chunks([sample])
            assert emitted == sample, sample
            assert state.leaked is False, sample

    def test_chunk_boundaries_preserve_bytes_without_canary(self) -> None:
        data = bytes((i * 31 + 7) % 256 for i in range(512))
        chunks = [data[i : i + 7] for i in range(0, len(data), 7)]
        emitted, state = scan_chunks(chunks)
        assert emitted == data
        assert state.leaked is False

    def test_partial_match_at_stream_end_is_flushed_verbatim(self) -> None:
        state = CanaryScanState(token=TOKEN)
        out = state.process(b"data " + NEEDLE[:9])
        assert out == b"data "
        assert state.flush() == NEEDLE[:9]
        assert state.flush() == b""  # idempotent

    def test_holdback_bounded_by_canary_length_minus_one(self) -> None:
        # A trailing partial match is held back, and only it: flush returns
        # exactly those bytes (9 < CANARY_LENGTH - 1).
        state = CanaryScanState(token=TOKEN)
        state.process(b"x" * 100 + NEEDLE[:9])
        assert state.flush() == NEEDLE[:9]

    def test_action_validation(self) -> None:
        with pytest.raises(ValueError):
            CanaryScanState(token=TOKEN, action="delete")


class TestScanStateBlock:
    def test_block_mode_drops_leak_chunk_and_terminates(self) -> None:
        state = CanaryScanState(token=TOKEN, action="block")
        assert state.process(b"safe first chunk") == b"safe first chunk"
        # Clean prefix before the match is emitted; leak onward is dropped.
        assert state.process(b"leak " + NEEDLE + b" more") == b"leak "
        assert state.leaked is True
        assert state.process(b"after") == b""  # terminal: always empty
        assert state.flush() == b""

    def test_block_mode_clean_stream_passthrough(self) -> None:
        emitted, state = scan_chunks([b"clean one", b"clean two"], action="block")
        assert emitted == b"clean oneclean two"
        assert state.leaked is False

    def test_alert_logged_once_with_fingerprint_not_value(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="lmpi.canary"):
            scan_chunks([b"a", NEEDLE, NEEDLE])
        records = [r for r in caplog.records if r.name == "lmpi.canary"]
        assert len(records) == 1
        event = json.loads(records[0].getMessage().split("detection event: ", 1)[1])
        assert event["stage"] == "canary"
        assert event["action"] == "redact"
        assert event["fingerprint"] == TOKEN.fingerprint
        assert event["occurrences"] == 1
        # The raw canary value never appears in logs.
        assert TOKEN.value not in records[0].getMessage()


# ---------------------------------------------------------------------------
# CanaryManager.scan_bytes — non-streaming response bodies
# ---------------------------------------------------------------------------


class TestScanBytes:
    def test_redacted_body_and_alert(self) -> None:
        manager = make_manager()
        payload = system_payload("sys")
        _, token = manager.inject(payload)
        assert token is not None
        body = json.dumps(
            {"choices": [{"message": {"content": f"leak {token.value} leak"}}]}
        ).encode("utf-8")

        out, state = manager.scan_bytes(body, token)
        assert token.value.encode("utf-8") not in out
        assert b"[REDACTED]" in out
        assert state.leaked is True
        # Body stays valid JSON after byte-level redaction.
        assert json.loads(out)["choices"][0]["message"]["content"] == (
            "leak [REDACTED] leak"
        )

    def test_clean_body_unchanged(self) -> None:
        manager = make_manager()
        payload = system_payload("sys")
        _, token = manager.inject(payload)
        assert token is not None
        body = b'{"ok": true, "content": "no canary here"}'
        out, state = manager.scan_bytes(body, token)
        assert out == body
        assert state.leaked is False

    def test_multiple_occurrences_all_redacted(self) -> None:
        manager = make_manager()
        payload = system_payload("sys")
        _, token = manager.inject(payload)
        assert token is not None
        body = f"{token.value}|{token.value}".encode("utf-8")
        out, state = manager.scan_bytes(body, token)
        assert out == b"[REDACTED]|[REDACTED]"
        assert state.occurrences == 2

    def test_scan_bytes_action_follows_settings(self) -> None:
        manager = CanaryManager(
            CanarySettings(secret="test-secret", action="block")
        )
        payload = system_payload("sys")
        _, token = manager.inject(payload)
        assert token is not None
        body = f"leak {token.value}".encode("utf-8")
        out, state = manager.scan_bytes(body, token)
        assert out == b"leak "  # clean prefix only; proxy returns 502 on leaked
        assert state.leaked is True
        assert state.action == "block"
