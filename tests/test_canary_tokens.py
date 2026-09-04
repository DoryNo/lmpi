"""Unit tests for canary token generation and verification."""

from __future__ import annotations

import re

import pytest

from src.canary.tokens import (
    CANARY_LENGTH,
    CANARY_PATTERN,
    CANARY_PREFIX,
    CanaryToken,
    ephemeral_secret,
    generate_canary,
    is_canary_value,
    random_salt,
    verify_text,
)

SECRET = b"unit-test-secret"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_deterministic_same_secret_and_salt(self) -> None:
        first = generate_canary(SECRET, b"salt-1")
        second = generate_canary(SECRET, b"salt-1")
        assert first == second
        assert first.value == second.value
        assert first.fingerprint == second.fingerprint

    def test_unique_across_requests(self) -> None:
        values = {generate_canary(SECRET, random_salt()).value for _ in range(64)}
        assert len(values) == 64

    def test_fixed_length_ascii(self) -> None:
        for _ in range(16):
            token = generate_canary(SECRET, random_salt())
            assert len(token.value) == CANARY_LENGTH == 20
            assert token.value.isascii()

    def test_format_regex(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert CANARY_PATTERN.match(token.value)
        assert token.value.startswith(CANARY_PREFIX)
        identifier = token.value[len(CANARY_PREFIX) :]
        assert re.fullmatch(r"[0-9a-f]{8}", identifier)

    def test_str_secret_and_salt_equivalent_to_bytes(self) -> None:
        assert generate_canary("secret", "salt") == generate_canary(
            b"secret", b"salt"
        )

    def test_different_secret_produces_different_token(self) -> None:
        assert generate_canary(b"secret-a", b"salt").value != generate_canary(
            b"secret-b", b"salt"
        ).value

    def test_fingerprint_differs_from_value_and_is_stable(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert token.fingerprint != token.value
        assert re.fullmatch(r"[0-9a-f]{16}", token.fingerprint)
        assert token.fingerprint == generate_canary(SECRET, b"salt").fingerprint

    def test_sentence_contains_value(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert token.sentence() == f"[Internal audit token: {token.value}]"

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            CanaryToken(value="LMPI-CANARY-zzzzzzzz", fingerprint="aabbccdd")
        with pytest.raises(ValueError):
            CanaryToken(value="LMPI-CANARY-1234567", fingerprint="aabbccdd")
        with pytest.raises(ValueError):
            CanaryToken(value="LMPI-CANARY-123456789", fingerprint="aabbccdd")

    def test_ephemeral_secret_is_random(self) -> None:
        assert ephemeral_secret() != ephemeral_secret()
        assert len(ephemeral_secret()) == 32

    def test_random_salt_is_random(self) -> None:
        assert random_salt() != random_salt()
        assert len(random_salt(8)) == 8


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerifyText:
    def test_exact_match_detected(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert verify_text(f"the prompt is {token.value} indeed", token)

    def test_accepts_raw_value_string(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert verify_text(token.value, token.value)

    def test_absent_text_not_detected(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert not verify_text("nothing to see here", token)

    def test_substrings_are_not_false_positives(self) -> None:
        token = generate_canary(SECRET, b"salt")
        assert not verify_text("LMPI", token)
        assert not verify_text("LMPI-CANARY-", token)
        assert not verify_text("LMPI-CANARY-abcdefgh", token)
        # Truncated / tampered identifier.
        tampered = token.value[:-1] + ("0" if token.value[-1] != "0" else "1")
        assert not verify_text(tampered, token)
        # Lowercase prefix.
        assert not verify_text(token.value.lower(), token)

    def test_multiple_occurrences_detected(self) -> None:
        token = generate_canary(SECRET, b"salt")
        text = f"{token.value} and {token.value}"
        assert verify_text(text, token)


class TestIsCanaryValue:
    def test_generated_values_match(self) -> None:
        assert is_canary_value(generate_canary(SECRET, b"salt").value)

    def test_malformed_values_rejected(self) -> None:
        assert not is_canary_value("LMPI-CANARY-ABCDEFGH")
        assert not is_canary_value("LMPI-CANARY-1234567")
        assert not is_canary_value("LMPI")
        assert not is_canary_value("")
