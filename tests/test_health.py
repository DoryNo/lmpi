"""Tests for GET /health (upstream is mocked, never the real network)."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app

UPSTREAM = "https://upstream.test"


def make_client(handler) -> TestClient:
    app = create_app(
        settings=Settings(upstream_url=UPSTREAM),
        transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_health_ok_when_upstream_reachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200)

    with make_client(handler) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"]
    assert data["upstream"]["url"] == UPSTREAM
    assert data["upstream"]["reachable"] is True
    assert data["upstream"]["status_code"] == 200


def test_health_still_ok_when_upstream_connection_fails() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with make_client(handler) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["upstream"]["reachable"] is False
    assert data["upstream"]["error"] == "ConnectError"


def test_health_reports_timeout_as_unreachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream too slow")

    with make_client(handler) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["upstream"]["reachable"] is False
    assert data["upstream"]["error"] == "ReadTimeout"
