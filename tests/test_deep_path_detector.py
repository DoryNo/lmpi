"""Detector-level tests for the deep path (stage 3) — pure stubs, no I/O."""

from __future__ import annotations

import math

import pytest

from src.deep_path import DeepPathDetector, StubBackend, decide_action
from src.deep_path.detector import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_MAX_CHARS,
    DEFAULT_WARN_THRESHOLD,
)


def make_detector(injection_prob: float, **kwargs) -> tuple[DeepPathDetector, StubBackend]:
    stub = StubBackend(scores=(1.0 - injection_prob, injection_prob))
    return DeepPathDetector(stub, **kwargs), stub


# ---------------------------------------------------------------------------
# Decision mapping (mirrors fast-path semantics)
# ---------------------------------------------------------------------------


class TestDecisionMapping:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, "block"),
            (0.75, "block"),  # exactly at threshold → block
            (0.7499, "warn"),
            (0.5, "warn"),  # exactly at warn → warn
            (0.4999, "allow"),
            (0.0, "allow"),
        ],
    )
    def test_default_thresholds(self, score: float, expected: str) -> None:
        assert decide_action(score, 0.75, 0.5) == expected

    def test_detector_block_boundary(self) -> None:
        detector, _ = make_detector(0.75)
        assert detector.detect("text").action == "block"

    def test_detector_warn_boundary(self) -> None:
        detector, _ = make_detector(0.7499)
        assert detector.detect("text").action == "warn"

    def test_detector_allow_below_warn(self) -> None:
        detector, _ = make_detector(0.1)
        assert detector.detect("text").action == "allow"

    def test_custom_thresholds(self) -> None:
        detector, _ = make_detector(0.6, block_threshold=0.5, warn_threshold=0.3)
        assert detector.detect("text").action == "block"

    def test_score_is_injection_probability(self) -> None:
        detector, _ = make_detector(0.83)
        result = detector.detect("text")
        assert math.isclose(result.score, 0.83)

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            DeepPathDetector(StubBackend(), warn_threshold=0.9, block_threshold=0.5)
        with pytest.raises(ValueError):
            DeepPathDetector(StubBackend(), block_threshold=1.5)
        with pytest.raises(ValueError):
            DeepPathDetector(StubBackend(), warn_threshold=-0.1)

    def test_invalid_max_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            DeepPathDetector(StubBackend(), max_chars=0)
        with pytest.raises(ValueError):
            DeepPathDetector(StubBackend(), max_chars=-5)


# ---------------------------------------------------------------------------
# Graceful degradation (no backend)
# ---------------------------------------------------------------------------


class TestUnavailable:
    def test_available_property(self) -> None:
        assert DeepPathDetector().available is False
        assert DeepPathDetector(StubBackend()).available is True

    def test_detect_without_backend_degrades_to_allow(self) -> None:
        result = DeepPathDetector().detect("anything")
        assert result.available is False
        assert result.action == "allow"
        assert result.score == 0.0

    def test_unavailable_reason_and_log(self) -> None:
        result = DeepPathDetector().detect("anything")
        assert "unavailable" in result.reason
        event = result.log_dict()
        assert event["available"] is False
        assert event["stage"] == "deep_path"


# ---------------------------------------------------------------------------
# Input hygiene: char cap on the classified text
# ---------------------------------------------------------------------------


class TestInputHygiene:
    def test_long_text_truncated_before_classification(self) -> None:
        detector, stub = make_detector(0.9, max_chars=10)
        result = detector.detect("x" * 100)
        assert stub.seen_texts == [["x" * 10]]
        assert result.char_truncated is True
        assert result.max_chars == 10

    def test_short_text_untouched(self) -> None:
        detector, stub = make_detector(0.9, max_chars=100)
        result = detector.detect("short")
        assert stub.seen_texts == [["short"]]
        assert result.char_truncated is False

    def test_default_max_chars_constant(self) -> None:
        assert DEFAULT_MAX_CHARS == 6000
        detector, _ = make_detector(0.1)
        assert detector.max_chars == 6000


# ---------------------------------------------------------------------------
# Latency + structured logging
# ---------------------------------------------------------------------------


class TestLatencyAndLogging:
    def test_latency_recorded_and_non_negative(self) -> None:
        detector, _ = make_detector(0.5)
        result = detector.detect("text")
        assert result.latency_ms >= 0.0
        assert result.log_dict()["latency_ms"] >= 0.0

    def test_log_dict_structure(self) -> None:
        detector, _ = make_detector(0.8)
        result = detector.detect("text")
        event = result.log_dict()
        assert event["stage"] == "deep_path"
        assert event["action"] == "block"
        assert event["score"] == 0.8
        assert event["model"] == "stub"
        assert event["quantized"] is False
        assert event["thresholds"] == {
            "block": DEFAULT_BLOCK_THRESHOLD,
            "warn": DEFAULT_WARN_THRESHOLD,
        }
        assert event["input_truncated_chars"] is False
        assert event["max_chars"] == 6000
        assert isinstance(event["latency_ms"], float)
        # JSON-serializable end to end
        import json

        json.dumps(event)

    def test_reason_includes_model_and_thresholds(self) -> None:
        detector, _ = make_detector(0.8)
        reason = detector.detect("text").reason
        assert "Deep path score=0.80" in reason
        assert "model=stub" in reason
        assert "block=0.75" in reason

    def test_result_carries_backend_metadata(self) -> None:
        class _Backend(StubBackend):
            model_name = "my-model"  # type: ignore[misc]

        detector = DeepPathDetector(_Backend())
        result = detector.detect("text")
        assert result.model == "my-model"
        assert result.quantized is False
