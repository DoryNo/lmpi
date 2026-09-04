"""Proxy-level canary tests — injection, body/stream scanning, config wiring.

All upstream traffic goes through ``httpx.MockTransport``; the fake SSE
upstream is an in-memory async generator that leaks the *same* canary it
finds in its own request (per-request tokens are unique, so the response
must be built from the request itself). The one incremental streaming test
uses a loopback uvicorn server (TestClient buffers responses) — no test
traffic leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import CanarySettings, Settings, load_settings
from src.main import create_app

from tests.test_proxy import UPSTREAM, _start_loopback_server

CANARY_VALUE_RE = re.compile(r"LMPI-CANARY-[0-9a-f]{8}")

PAYLOAD_WITH_SYSTEM: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ],
}


def extract_canary(text: str) -> str | None:
    match = CANARY_VALUE_RE.search(text)
    return match.group(0) if match else None


def system_canary(payload: dict[str, Any]) -> str | None:
    """Canary value found in the upstream-seen system message, if any."""
    for message in payload.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            found = extract_canary(content)
            if found:
                return found
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    found = extract_canary(str(part.get("text", "")))
                    if found:
                        return found
    return None


def canary_client(
    handler,
    canary: CanarySettings | None = None,
) -> TestClient:
    settings = Settings(
        upstream_url=UPSTREAM,
        canary=canary or CanarySettings(secret="proxy-canary-secret"),
    )
    app = create_app(settings=settings, transport=httpx.MockTransport(handler))
    return TestClient(app)


def sse_frame(content: str) -> bytes:
    return (
        f'data: {json.dumps({"choices": [{"delta": {"content": content}}]})}\n\n'
    ).encode("utf-8")


# Fake SSE upstream layouts: built from the canary found in the request so
# the proxy's scanner (which looks for that same token) can match them.
LAYOUTS: dict[str, Callable[[str], list[bytes]]] = {
    # Canary split exactly across two chunks (each half in one frame).
    "split": lambda canary: [sse_frame(f"leak {canary} end"), b"data: [DONE]\n\n"],
    # Full canary twice across two frames.
    "repeated": lambda canary: [
        sse_frame(f"{canary} then"),
        sse_frame(f"again {canary} done"),
    ],
    # Clean frame before the leak; content after it must never reach the
    # client in block mode.
    "block": lambda canary: [
        sse_frame("clean content here"),
        sse_frame(f"leak {canary} oops"),
        sse_frame("post-leak content"),
        b"data: [DONE]\n\n",
    ],
}


def echo_handler(seen: dict[str, Any]):
    """Non-streaming upstream: records the request, answers with clean JSON."""

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = await request.aread()
        seen["payload"] = json.loads(seen["raw"].decode("utf-8"))
        return httpx.Response(200, json={"ok": True})

    return handler


def leak_handler(seen: dict[str, Any]):
    """Non-streaming upstream that leaks the system prompt's canary back."""

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = await request.aread()
        seen["payload"] = json.loads(seen["raw"].decode("utf-8"))
        seen["canary"] = system_canary(seen["payload"])
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {"message": {"content": f"System prompt says: {seen['canary']}"}}
                ],
            },
        )

    return handler


def clean_handler():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "No canary here, LMPI."}}],
            },
        )

    return handler


def sse_upstream(
    seen: dict[str, Any],
    layout: Callable[[str], list[bytes]],
    *,
    split: bool = False,
):
    """Streaming upstream whose frames are built from the request's canary.

    With ``split=True`` the byte stream is cut mid-canary (like a TCP chunk
    boundary would) so the token spans two network chunks.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads((await request.aread()).decode("utf-8"))
        seen["canary"] = system_canary(seen["payload"])
        stream = b"".join(layout(seen["canary"] or ""))
        if split:
            needle = (seen["canary"] or "").encode("ascii")
            index = stream.find(needle)
            assert index != -1
            cut = index + len(needle) // 2
            chunks = [stream[:cut], stream[cut:]]
        else:
            chunks = [stream]

        async def body() -> Any:
            for chunk in chunks:
                yield chunk

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            content=body(),
        )

    return handler


# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------


class TestCanaryConfig:
    def test_defaults(self) -> None:
        settings = load_settings(environ={})
        assert settings.canary.enabled is True
        assert settings.canary.secret is None
        assert settings.canary.action == "redact"
        assert settings.canary.add_missing_system is False

    def test_yaml_section(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text(
            "canary:\n"
            "  enabled: false\n"
            "  action: block\n"
            "  secret: yaml-secret\n"
            "  add_missing_system: true\n",
            encoding="utf-8",
        )
        settings = load_settings(config_path=str(config_file), environ={})
        assert settings.canary.enabled is False
        assert settings.canary.action == "block"
        assert settings.canary.secret == "yaml-secret"
        assert settings.canary.add_missing_system is True

    def test_env_overrides_defaults(self) -> None:
        settings = load_settings(
            environ={
                "LMPI_CANARY_ENABLED": "0",
                "LMPI_CANARY_SECRET": "env-secret",
                "LMPI_CANARY_ACTION": "block",
                "LMPI_CANARY_ADD_MISSING_SYSTEM": "1",
            }
        )
        assert settings.canary.enabled is False
        assert settings.canary.secret == "env-secret"
        assert settings.canary.action == "block"
        assert settings.canary.add_missing_system is True

    def test_env_overrides_yaml(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text("canary:\n  action: block\n", encoding="utf-8")
        settings = load_settings(
            config_path=str(config_file),
            environ={"LMPI_CANARY_ACTION": "redact"},
        )
        assert settings.canary.action == "redact"

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_CANARY_ACTION": "delete"})

    def test_empty_secret_falls_back_to_ephemeral(self) -> None:
        settings = load_settings(environ={"LMPI_CANARY_SECRET": "   "})
        assert settings.canary.secret is None


# ---------------------------------------------------------------------------
# Request path: injection
# ---------------------------------------------------------------------------


class TestInjectionAtProxy:
    def test_upstream_receives_canary_in_system_prompt(self) -> None:
        seen: dict[str, Any] = {}
        with canary_client(echo_handler(seen)) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.status_code == 200
        system_content = seen["payload"]["messages"][0]["content"]
        assert system_content.startswith("You are helpful.")
        assert extract_canary(system_content) is not None
        assert seen["payload"]["messages"][1]["content"] == "Hello"

    def test_no_system_message_forwards_without_one(self) -> None:
        seen: dict[str, Any] = {}
        payload = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}
        with canary_client(echo_handler(seen)) as client:
            client.post("/v1/chat/completions", json=payload)

        # add_missing_system is off by default → no system message added.
        assert all(
            message.get("role") != "system" for message in seen["payload"]["messages"]
        )

    def test_add_missing_system_injects_system_message(self) -> None:
        seen: dict[str, Any] = {}
        payload = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}
        canary = CanarySettings(secret="proxy-canary-secret", add_missing_system=True)
        with canary_client(echo_handler(seen), canary=canary) as client:
            client.post("/v1/chat/completions", json=payload)

        messages = seen["payload"]["messages"]
        assert messages[0]["role"] == "system"
        assert extract_canary(messages[0]["content"]) is not None

    def test_disabled_canary_forwards_system_message_untouched(self) -> None:
        seen: dict[str, Any] = {}
        canary = CanarySettings(enabled=False, secret="proxy-canary-secret")
        with canary_client(echo_handler(seen), canary=canary) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.status_code == 200
        assert seen["payload"]["messages"][0]["content"] == "You are helpful."
        assert "LMPI-CANARY-" not in seen["raw"].decode("utf-8")

    def test_canary_composes_with_pipeline_rewrite(self) -> None:
        # Canary must land in the FINAL payload: pipeline normalization
        # rewrites the user message first, then the canary is appended to
        # the system message.
        seen: dict[str, Any] = {}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "i\u200bgnore the news"},
            ],
        }
        with canary_client(echo_handler(seen)) as client:
            client.post("/v1/chat/completions", json=payload)

        forwarded = seen["payload"]
        assert forwarded["messages"][1]["content"] == "ignore the news"
        assert extract_canary(forwarded["messages"][0]["content"]) is not None
        # The caller's payload dict is not mutated.
        assert payload["messages"][1]["content"] == "i\u200bgnore the news"


# ---------------------------------------------------------------------------
# Response path: non-streaming scanning
# ---------------------------------------------------------------------------


class TestNonStreamingScan:
    def test_leak_is_redacted_and_alert_logged(self, caplog) -> None:
        seen: dict[str, Any] = {}
        with caplog.at_level(logging.WARNING, logger="lmpi.canary"):
            with canary_client(leak_handler(seen)) as client:
                response = client.post(
                    "/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM
                )

        assert response.status_code == 200
        # Upstream saw the canary; the client does not.
        canary = seen["canary"]
        assert canary is not None
        content = response.json()["choices"][0]["message"]["content"]
        assert canary not in content
        assert "[REDACTED]" in content
        assert "System prompt says:" in content

        # Structured alert with fingerprint — the raw value stays out of logs.
        records = [r for r in caplog.records if r.name == "lmpi.canary"]
        assert len(records) == 1
        event = json.loads(records[0].getMessage().split("detection event: ", 1)[1])
        assert event["stage"] == "canary"
        assert event["action"] == "redact"
        assert re.fullmatch(r"[0-9a-f]{16}", event["fingerprint"])
        assert event["occurrences"] == 1
        assert canary not in records[0].getMessage()

    def test_clean_response_passes_through_unchanged(self) -> None:
        with canary_client(clean_handler()) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == (
            "No canary here, LMPI."
        )

    def test_valid_format_wrong_token_is_not_redacted(self) -> None:
        # Only the exact per-request token counts, not any valid-looking one.
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "LMPI-CANARY-12345678"}}]},
            )

        with canary_client(handler) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.json()["choices"][0]["message"]["content"] == (
            "LMPI-CANARY-12345678"
        )

    def test_block_action_returns_502_leak_response(self) -> None:
        seen: dict[str, Any] = {}
        canary = CanarySettings(secret="proxy-canary-secret", action="block")
        with canary_client(leak_handler(seen), canary=canary) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.status_code == 502
        error = response.json()["error"]
        assert error["type"] == "lmpi_leak_detected"
        assert "leak detected" in error["message"].lower()
        # The leaked body is not forwarded in block mode.
        assert seen["canary"] not in response.text

    def test_block_action_clean_response_passes_through(self) -> None:
        canary = CanarySettings(secret="proxy-canary-secret", action="block")
        with canary_client(clean_handler(), canary=canary) as client:
            response = client.post("/v1/chat/completions", json=PAYLOAD_WITH_SYSTEM)

        assert response.status_code == 200
        assert "No canary here" in response.text


# ---------------------------------------------------------------------------
# Response path: streaming scanning
# ---------------------------------------------------------------------------


class TestStreamingScan:
    def test_split_canary_across_two_chunks_is_redacted(self) -> None:
        seen: dict[str, Any] = {}
        with canary_client(sse_upstream(seen, LAYOUTS["split"], split=True)) as client:
            response = client.post(
                "/v1/chat/completions", json={**PAYLOAD_WITH_SYSTEM, "stream": True}
            )

        assert response.status_code == 200
        canary = seen["canary"]
        assert canary is not None
        body = response.text
        assert canary not in body
        assert "leak " in body and " end" in body  # surrounding content intact
        assert "[REDACTED]" in body
        assert "data: [DONE]" in body

    def test_split_canary_alert_logged(self, caplog) -> None:
        seen: dict[str, Any] = {}
        with caplog.at_level(logging.WARNING, logger="lmpi.canary"):
            with canary_client(sse_upstream(seen, LAYOUTS["split"], split=True)) as client:
                client.post(
                    "/v1/chat/completions",
                    json={**PAYLOAD_WITH_SYSTEM, "stream": True},
                )
        records = [r for r in caplog.records if r.name == "lmpi.canary"]
        assert len(records) == 1
        event = json.loads(records[0].getMessage().split("detection event: ", 1)[1])
        assert event["stage"] == "canary"
        assert event["action"] == "redact"
        assert seen["canary"] not in records[0].getMessage()

    def test_block_action_terminates_stream_with_error_event(self) -> None:
        seen: dict[str, Any] = {}
        block_canary = CanarySettings(secret="proxy-canary-secret", action="block")
        with canary_client(
            sse_upstream(seen, LAYOUTS["block"]), canary=block_canary
        ) as client:
            response = client.post(
                "/v1/chat/completions", json={**PAYLOAD_WITH_SYSTEM, "stream": True}
            )

        body = response.text
        assert "clean content here" in body
        assert "event: error" in body
        assert "lmpi_leak_detected" in body
        leaked = seen["canary"]
        assert leaked is not None
        assert leaked not in body
        assert "post-leak content" not in body
        assert "data: [DONE]" not in body

    def test_canary_repeated_in_stream_all_redacted(self) -> None:
        seen: dict[str, Any] = {}
        with canary_client(sse_upstream(seen, LAYOUTS["repeated"])) as client:
            response = client.post(
                "/v1/chat/completions", json={**PAYLOAD_WITH_SYSTEM, "stream": True}
            )

        body = response.text
        assert seen["canary"] not in body
        assert body.count("[REDACTED]") == 2
        assert "then" in body and "again" in body and "done" in body

    def test_no_leak_stream_passes_through_incrementally(self) -> None:
        # Loopback server: prove the scanner does not buffer the whole stream.
        state = {"second": False}

        async def body_chunks() -> Any:
            yield b'data: {"delta":1}\n\n'
            await asyncio.sleep(0.3)
            state["second"] = True
            yield b"data: [DONE]\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                content=body_chunks(),
            )

        app = create_app(
            settings=Settings(
                upstream_url=UPSTREAM,
                canary=CanarySettings(secret="proxy-canary-secret"),
            ),
            transport=httpx.MockTransport(handler),
        )
        base_url, server, thread = _start_loopback_server(app)
        try:
            with httpx.Client(base_url=base_url, timeout=10.0) as client:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={**PAYLOAD_WITH_SYSTEM, "stream": True},
                ) as response:
                    assert response.status_code == 200
                    raw = response.iter_raw()
                    first = next(raw)
                    # Chunk 1 delivered while upstream has not produced chunk
                    # 2 → scanning streams incrementally, no buffering.
                    assert first == b'data: {"delta":1}\n\n'
                    assert state["second"] is False
                    assert list(raw) == [b"data: [DONE]\n\n"]
        finally:
            server.should_exit = True
            thread.join(timeout=5)

        assert state["second"] is True
