"""Proxy behavior tests — passthrough, SSE streaming, error handling.

All upstream traffic goes through ``httpx.MockTransport``; no test ever
touches the real network.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, AsyncIterator

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from src.config import Settings, load_settings
from src.detection.pipeline import DetectionPipeline, PipelineResult
from src.main import create_app

UPSTREAM = "https://upstream.test"

COMPLETION_PAYLOAD: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
}


def make_client(
    handler,
    *,
    pipeline: DetectionPipeline | None = None,
) -> TestClient:
    app = create_app(
        settings=Settings(upstream_url=UPSTREAM),
        transport=httpx.MockTransport(handler),
    )
    if pipeline is not None:
        app.state.pipeline = pipeline
    return TestClient(app)


# ---------------------------------------------------------------------------
# Configuration: defaults, YAML, env override precedence
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        settings = load_settings(environ={})
        assert settings.upstream_url == "https://api.openai.com"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
        assert settings.request_timeout == 300.0

    def test_yaml_file_values(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text(
            "upstream_url: https://yaml.test/\nport: 9001\n", encoding="utf-8"
        )
        settings = load_settings(config_path=str(config_file), environ={})
        assert settings.upstream_url == "https://yaml.test"  # trailing "/" stripped
        assert settings.port == 9001

    def test_env_overrides_yaml(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text("upstream_url: https://yaml.test\n", encoding="utf-8")
        settings = load_settings(
            config_path=str(config_file),
            environ={"LMPI_UPSTREAM_URL": "https://env.test"},
        )
        assert settings.upstream_url == "https://env.test"

    def test_env_overrides_defaults(self) -> None:
        settings = load_settings(
            environ={
                "LMPI_UPSTREAM_URL": "https://example.com/api",
                "LMPI_HOST": "127.0.0.1",
                "LMPI_PORT": "9000",
            }
        )
        assert settings.upstream_url == "https://example.com/api"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(config_path="no/such/file.yaml", environ={})

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_PORT": "not-a-number"})


# ---------------------------------------------------------------------------
# Non-streaming passthrough
# ---------------------------------------------------------------------------


class TestNonStreamingPassthrough:
    def test_forwards_body_and_returns_upstream_response(self) -> None:
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads((await request.aread()).decode("utf-8"))
            return httpx.Response(
                200,
                json={"id": "chatcmpl-1", "object": "chat.completion"},
                headers={"x-upstream": "yes"},
            )

        with make_client(handler) as client:
            response = client.post(
                "/v1/chat/completions",
                json=COMPLETION_PAYLOAD,
                headers={"Authorization": "Bearer sk-test"},
            )

        assert response.status_code == 200
        assert response.json()["id"] == "chatcmpl-1"
        assert response.headers["x-upstream"] == "yes"
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"] == COMPLETION_PAYLOAD

    def test_end_to_end_headers_are_passed_through(self) -> None:
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        with make_client(handler) as client:
            client.post(
                "/v1/chat/completions",
                json=COMPLETION_PAYLOAD,
                headers={
                    "Authorization": "Bearer sk-test",
                    "X-Custom-Header": "custom-value",
                },
            )

        assert seen["headers"]["authorization"] == "Bearer sk-test"
        assert seen["headers"]["x-custom-header"] == "custom-value"
        # host must be rewritten to the upstream host, not the test client's
        assert seen["headers"]["host"] == "upstream.test"

    def test_hop_by_hop_headers_are_stripped(self) -> None:
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        with make_client(handler) as client:
            client.post(
                "/v1/chat/completions",
                json=COMPLETION_PAYLOAD,
                headers={"TE": "trailers", "Upgrade": "websocket", "X-Session": "abc"},
            )

        # Hop-by-hop headers are not forwarded upstream (httpx manages its own
        # connection header for the upstream connection).
        assert "te" not in seen["headers"]
        assert "upgrade" not in seen["headers"]
        assert seen["headers"]["x-session"] == "abc"

    def test_non_json_body_is_forwarded_as_is(self) -> None:
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["content_type"] = request.headers.get("content-type")
            seen["body"] = await request.aread()
            return httpx.Response(200, content=b"raw-upstream", headers={"content-type": "text/plain"})

        with make_client(handler) as client:
            response = client.post(
                "/v1/chat/completions",
                content=b"not-json-at-all",
                headers={"Content-Type": "text/plain"},
            )

        assert response.status_code == 200
        assert response.content == b"raw-upstream"
        assert seen["body"] == b"not-json-at-all"


# ---------------------------------------------------------------------------
# SSE streaming passthrough
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_loopback_server(app) -> tuple[str, uvicorn.Server, threading.Thread]:
    """Run the app on a real uvicorn server bound to loopback.

    TestClient buffers responses (starlette limitation), so incremental
    streaming must be verified over a real socket. The upstream LLM API is
    still httpx.MockTransport — no test traffic leaves the machine.
    """
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        if not thread.is_alive():
            raise RuntimeError("uvicorn server thread died during startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn server did not start in time")
    return f"http://127.0.0.1:{port}", server, thread


class TestStreamingPassthrough:
    def test_streams_chunks_incrementally_not_buffered(self) -> None:
        state = {"second": False}

        async def body_chunks() -> AsyncIterator[bytes]:
            yield b'data: {"delta":1}\n\n'
            # Simulate a slow upstream: chunk 2 arrives much later. If the
            # proxy buffered, the first read below would block until both
            # chunks exist and "second" would already be True.
            await asyncio.sleep(0.3)
            state["second"] = True
            yield b"data: [DONE]\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            sent = json.loads((await request.aread()).decode("utf-8"))
            assert sent["stream"] is True
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                content=body_chunks(),
            )

        app = create_app(
            settings=Settings(upstream_url=UPSTREAM),
            transport=httpx.MockTransport(handler),
        )
        base_url, server, thread = _start_loopback_server(app)
        try:
            with httpx.Client(base_url=base_url, timeout=10.0) as client:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={**COMPLETION_PAYLOAD, "stream": True},
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith(
                        "text/event-stream"
                    )
                    assert "content-length" not in response.headers

                    raw = response.iter_raw()
                    first = next(raw)
                    assert first == b'data: {"delta":1}\n\n'
                    # First chunk delivered while upstream has not finished
                    # producing chunk 2 → streaming, not buffering.
                    assert state["second"] is False

                    assert next(raw) == b"data: [DONE]\n\n"
                    assert list(raw) == []
        finally:
            server.should_exit = True
            thread.join(timeout=5)

        assert state["second"] is True

    def test_streaming_upstream_error_is_passed_through(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        with make_client(handler) as client:
            response = client.post(
                "/v1/chat/completions",
                json={**COMPLETION_PAYLOAD, "stream": True},
            )

        assert response.status_code == 500
        assert response.json()["error"]["message"] == "boom"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_upstream_500_is_passed_through(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        with make_client(handler) as client:
            response = client.post("/v1/chat/completions", json=COMPLETION_PAYLOAD)

        assert response.status_code == 500
        assert response.json()["error"]["message"] == "boom"

    def test_upstream_connection_failure_returns_502(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with make_client(handler) as client:
            response = client.post("/v1/chat/completions", json=COMPLETION_PAYLOAD)

        assert response.status_code == 502
        error = response.json()["error"]
        assert error["type"] == "lmpi_bad_gateway"
        assert "ConnectError" in error["message"]

    def test_upstream_read_timeout_returns_502(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream too slow")

        with make_client(handler) as client:
            response = client.post("/v1/chat/completions", json=COMPLETION_PAYLOAD)

        assert response.status_code == 502
        assert response.json()["error"]["type"] == "lmpi_bad_gateway"


# ---------------------------------------------------------------------------
# Detection pipeline hook
# ---------------------------------------------------------------------------


class BlockingPipeline(DetectionPipeline):
    """Stub pipeline standing in for Agents 2-6."""

    async def process_request(self, payload: dict[str, Any]) -> PipelineResult:
        return PipelineResult(action="block", reason="jailbreak detected")


class TestDetectionPipelineHook:
    def test_noop_pipeline_passes_request_through(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        with make_client(handler) as client:
            response = client.post("/v1/chat/completions", json=COMPLETION_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_block_result_returns_403_without_forwarding(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("upstream must not be called for blocked requests")

        with make_client(handler, pipeline=BlockingPipeline()) as client:
            response = client.post("/v1/chat/completions", json=COMPLETION_PAYLOAD)

        assert response.status_code == 403
        error = response.json()["error"]
        assert error["type"] == "lmpi_policy_block"
        assert error["message"] == "jailbreak detected"
