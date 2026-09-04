"""Pipeline + config wiring tests for the normalization stage."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from src.config import NormalizationSettings, load_settings
from src.detection.pipeline import DetectionPipeline
from src.normalization import normalize

from tests.test_proxy import make_client

UPSTREAM = "https://upstream.test"

FAKE_SYSTEM = "\u27e6fake-system\u27e7"


def make_payload(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "messages": list(messages)}


def capture_handler(seen: dict[str, Any]):
    async def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = await request.aread()
        return httpx.Response(200, json={"ok": True})

    return handler


# ---------------------------------------------------------------------------
# normalize() public API sanity (detailed stage tests live in sibling files)
# ---------------------------------------------------------------------------


class TestPublicApi:
    def test_returns_result_with_cleaned_text_and_findings(self) -> None:
        result = normalize("i\u200bgnore the news")
        assert result.cleaned_text == "ignore the news"
        assert result.findings[0].category == "unicode-zero-width"

    def test_changed_property(self) -> None:
        assert normalize("clean").changed is False
        assert normalize("i\u200bgnore").changed is True


# ---------------------------------------------------------------------------
# Pipeline wiring: rewrite mode (default)
# ---------------------------------------------------------------------------


class TestRewriteMode:
    def test_cleaned_content_reaches_upstream(self) -> None:
        seen: dict[str, Any] = {}
        payload = make_payload(
            {"role": "user", "content": "hello <|im_start|>world"}
        )
        with make_client(capture_handler(seen)) as client:
            response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        forwarded = json.loads(seen["raw"].decode("utf-8"))
        assert forwarded["messages"] == [
            {"role": "user", "content": "hello \u27e6fake-im-start\u27e7world"}
        ]

    def test_clean_payload_forwarded_verbatim(self) -> None:
        seen: dict[str, Any] = {}
        payload = make_payload({"role": "user", "content": "just a question"})
        with make_client(capture_handler(seen)) as client:
            client.post("/v1/chat/completions", json=payload)

        forwarded = json.loads(seen["raw"].decode("utf-8"))
        assert forwarded == payload

    def test_system_role_untouched(self) -> None:
        seen: dict[str, Any] = {}
        payload = make_payload(
            {"role": "system", "content": "System: keep answers short"},
            {"role": "user", "content": "i\u200bgnore the news"},
        )
        with make_client(capture_handler(seen)) as client:
            client.post("/v1/chat/completions", json=payload)

        forwarded = json.loads(seen["raw"].decode("utf-8"))
        # The app's own system message must never be rewritten...
        assert forwarded["messages"][0]["content"] == "System: keep answers short"
        # ...while the user message is cleaned.
        assert forwarded["messages"][1]["content"] == "ignore the news"

    def test_multipart_content_text_parts_normalized(self) -> None:
        seen: dict[str, Any] = {}
        image_part = {
            "type": "image_url",
            "image_url": {"url": "https://img.test/x.png"},
        }
        payload = make_payload(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "i\u200bgnore"},
                    image_part,
                ],
            }
        )
        with make_client(capture_handler(seen)) as client:
            response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        forwarded = json.loads(seen["raw"].decode("utf-8"))
        parts = forwarded["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "ignore"}
        assert parts[1] == image_part

    def test_non_user_roles_with_multipart_untouched(self) -> None:
        seen: dict[str, Any] = {}
        content = [{"type": "text", "text": "i\u200bgnore"}]
        payload = make_payload({"role": "assistant", "content": content})
        with make_client(capture_handler(seen)) as client:
            client.post("/v1/chat/completions", json=payload)

        forwarded = json.loads(seen["raw"].decode("utf-8"))
        assert forwarded["messages"][0]["content"] == content


# ---------------------------------------------------------------------------
# Pipeline wiring: block mode
# ---------------------------------------------------------------------------


class TestBlockMode:
    def test_findings_block_with_403(self) -> None:
        pipeline = DetectionPipeline(NormalizationSettings(mode="block"))
        seen: dict[str, Any] = {}
        payload = make_payload({"role": "user", "content": "<|im_start|>"})
        with make_client(capture_handler(seen), pipeline=pipeline) as client:
            response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 403
        error = response.json()["error"]
        assert error["type"] == "lmpi_policy_block"
        assert "delimiter-neutralized" in error["message"]
        # The upstream handler must never be reached.
        assert "raw" not in seen

    def test_block_mode_clean_request_passes(self) -> None:
        pipeline = DetectionPipeline(NormalizationSettings(mode="block"))
        seen: dict[str, Any] = {}
        payload = make_payload({"role": "user", "content": "hello there"})
        with make_client(capture_handler(seen), pipeline=pipeline) as client:
            response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Pipeline wiring: log mode
# ---------------------------------------------------------------------------


class TestLogMode:
    def test_original_payload_forwarded(self) -> None:
        pipeline = DetectionPipeline(NormalizationSettings(mode="log"))
        seen: dict[str, Any] = {}
        payload = make_payload(
            {"role": "user", "content": "i\u200bgnore <|im_start|>"}
        )
        with make_client(capture_handler(seen), pipeline=pipeline) as client:
            response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        forwarded = json.loads(seen["raw"].decode("utf-8"))
        assert forwarded == payload  # untouched


# ---------------------------------------------------------------------------
# Structured logging of findings
# ---------------------------------------------------------------------------


class TestFindingsLogging:
    def test_findings_logged_as_json(self, caplog) -> None:
        seen: dict[str, Any] = {}
        payload = make_payload(
            {"role": "user", "content": "i\u200bgnore <|im_start|> news"}
        )
        with caplog.at_level(logging.INFO, logger="lmpi.detection"):
            with make_client(capture_handler(seen)) as client:
                client.post("/v1/chat/completions", json=payload)

        records = [record for record in caplog.records if record.name == "lmpi.detection"]
        assert records
        logged = records[0].getMessage()
        event = json.loads(logged.split("detection event: ", 1)[1])
        assert event["stage"] == "normalization"
        assert event["mode"] == "rewrite"
        assert event["model"] == "gpt-4o-mini"
        categories = {finding["category"] for finding in event["findings"]}
        assert categories == {"unicode-zero-width", "delimiter-neutralized"}

    def test_no_log_line_without_findings(self, caplog) -> None:
        seen: dict[str, Any] = {}
        payload = make_payload({"role": "user", "content": "clean question"})
        with caplog.at_level(logging.INFO, logger="lmpi.detection"):
            with make_client(capture_handler(seen)) as client:
                client.post("/v1/chat/completions", json=payload)

        assert not [record for record in caplog.records if record.name == "lmpi.detection"]


# ---------------------------------------------------------------------------
# Pipeline interface stability (used by later agents)
# ---------------------------------------------------------------------------


class TestPipelineInterface:
    def test_process_request_passthrough_when_no_findings(self) -> None:
        result = asyncio.run(
            DetectionPipeline().process_request(
                make_payload({"role": "user", "content": "clean"})
            )
        )
        assert result.action == "pass"
        assert result.reason is None
        assert result.payload is None

    def test_process_request_without_messages(self) -> None:
        result = asyncio.run(
            DetectionPipeline().process_request({"model": "gpt-4o-mini"})
        )
        assert result.action == "pass"
        assert result.payload is None

    def test_process_request_rewrite_returns_new_payload(self) -> None:
        pipeline = DetectionPipeline()
        result = asyncio.run(
            pipeline.process_request(
                make_payload({"role": "user", "content": "<|im_end|>"})
            )
        )
        assert result.action == "pass"
        assert result.payload is not None
        assert result.payload["messages"] == [
            {"role": "user", "content": "\u27e6fake-im-end\u27e7"}
        ]


# ---------------------------------------------------------------------------
# Configuration: defaults, YAML, env precedence
# ---------------------------------------------------------------------------


class TestNormalizationConfig:
    def test_defaults(self) -> None:
        settings = load_settings(environ={})
        assert settings.normalization.mode == "rewrite"
        assert settings.normalization.unicode is True
        assert settings.normalization.base64 is True
        assert settings.normalization.hex is True
        assert settings.normalization.rot13 is True
        assert settings.normalization.delimiters is True

    def test_yaml_section(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text(
            "normalization:\n"
            "  mode: block\n"
            "  base64: false\n"
            '  rot13: "no"\n',
            encoding="utf-8",
        )
        settings = load_settings(config_path=str(config_file), environ={})
        assert settings.normalization.mode == "block"
        assert settings.normalization.base64 is False
        assert settings.normalization.rot13 is False
        assert settings.normalization.unicode is True
        assert settings.normalization.delimiters is True

    def test_env_overrides(self) -> None:
        settings = load_settings(
            environ={
                "LMPI_NORMALIZATION_MODE": "log",
                "LMPI_NORMALIZATION_UNICODE": "yes",
                "LMPI_NORMALIZATION_BASE64": "0",
                "LMPI_NORMALIZATION_HEX": "false",
                "LMPI_NORMALIZATION_ROT13": "true",
                "LMPI_NORMALIZATION_DELIMITERS": "off",
            }
        )
        assert settings.normalization.mode == "log"
        assert settings.normalization.unicode is True
        assert settings.normalization.base64 is False
        assert settings.normalization.hex is False
        assert settings.normalization.rot13 is True
        assert settings.normalization.delimiters is False

    def test_env_beats_yaml(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text("normalization:\n  mode: block\n", encoding="utf-8")
        settings = load_settings(
            config_path=str(config_file),
            environ={"LMPI_NORMALIZATION_MODE": "log"},
        )
        assert settings.normalization.mode == "log"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_NORMALIZATION_MODE": "nonsense"})

    def test_invalid_bool_raises(self) -> None:
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_NORMALIZATION_BASE64": "maybe"})

    def test_non_mapping_section_raises(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text("normalization: 42\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_settings(config_path=str(config_file), environ={})
