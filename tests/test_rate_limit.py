"""Rate limiting + body-size limit tests (v1.1 hardening).

Unit tests for the token bucket use an injected clock (no sleeping);
proxy integration tests run through TestClient with ``httpx.MockTransport``
— fully offline.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import (
    CanarySettings,
    RateLimitSettings,
    Settings,
    load_settings,
)
from src.main import create_app
from src.proxy import BodyTooLarge
from src.rate_limit import (
    MAX_BUCKETS,
    RateLimiter,
    TokenBucket,
    build_rate_limiter,
    client_key,
)

UPSTREAM = "https://upstream.test"

COMPLETION_PAYLOAD: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
}


def make_client(
    handler,
    *,
    rate_limit: RateLimitSettings | None = None,
    max_body_bytes: int | None = None,
) -> TestClient:
    settings = Settings(
        upstream_url=UPSTREAM,
        canary=CanarySettings(enabled=False, secret="test-secret"),
    )
    if rate_limit is not None:
        settings = Settings(
            upstream_url=UPSTREAM,
            canary=CanarySettings(enabled=False, secret="test-secret"),
            rate_limit=rate_limit,
        )
    if max_body_bytes is not None:
        settings = Settings(
            upstream_url=UPSTREAM,
            canary=CanarySettings(enabled=False, secret="test-secret"),
            rate_limit=rate_limit or RateLimitSettings(enabled=False),
            max_body_bytes=max_body_bytes,
        )
    app = create_app(
        settings=settings, transport=httpx.MockTransport(handler)
    )
    return TestClient(app)


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "chatcmpl-1"})


# ---------------------------------------------------------------------------
# Unit: TokenBucket refill math
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_starts_full_burst_allowed_then_denied(self) -> None:
        bucket = TokenBucket(capacity=3, refill_per_second=1.0, now=0.0)
        assert bucket.try_acquire(now=0.0) == (True, 0.0)
        assert bucket.try_acquire(now=0.0) == (True, 0.0)
        assert bucket.try_acquire(now=0.0) == (True, 0.0)
        allowed, retry_after = bucket.try_acquire(now=0.0)
        assert not allowed
        assert 0.5 <= retry_after <= 1.0
        assert bucket.remaining() == 0

    def test_refill_accrues_with_elapsed_time(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0, now=0.0)
        assert bucket.try_acquire(now=0.0)[0]
        assert bucket.try_acquire(now=0.0)[0]
        assert not bucket.try_acquire(now=0.0)[0]
        # 0.5s later: half a token — still not enough.
        assert not bucket.try_acquire(now=0.5)[0]
        # A full second later: one token refilled.
        allowed, retry_after = bucket.try_acquire(now=1.0)
        assert allowed
        assert retry_after == 0.0

    def test_refill_capped_at_capacity(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0, now=0.0)
        bucket.try_acquire(now=0.0)
        bucket.try_acquire(now=0.0)
        # A huge idle gap must not grant more than the burst capacity.
        allowed, _ = bucket.try_acquire(now=10_000.0)
        assert allowed
        allowed, _ = bucket.try_acquire(now=10_000.0)
        assert allowed
        assert not bucket.try_acquire(now=10_000.0)[0]
        assert bucket.remaining() == 0

    def test_remaining_tracks_partial_refill(self) -> None:
        bucket = TokenBucket(capacity=5, refill_per_second=2.0, now=0.0)
        for _ in range(5):
            bucket.try_acquire(now=0.0)
        assert bucket.remaining() == 0
        # 0.4s × 2 tokens/s = 0.8 tokens — not enough for one request.
        allowed, _ = bucket.try_acquire(now=0.4)
        assert not allowed
        assert bucket.remaining() == 0
        # 1.0s × 2 = 2 tokens — one request admitted, one whole token left.
        allowed, _ = bucket.try_acquire(now=1.0)
        assert allowed
        assert bucket.remaining() == 1

    def test_invalid_constructor_args_raise(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(capacity=0, refill_per_second=1.0, now=0.0)
        with pytest.raises(ValueError):
            TokenBucket(capacity=3, refill_per_second=0.0, now=0.0)


# ---------------------------------------------------------------------------
# Unit: RateLimiter per-key isolation + bounds
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_keys_are_isolated(self) -> None:
        limiter = RateLimiter(
            RateLimitSettings(enabled=True, requests_per_minute=60, burst=2),
            clock=lambda: 0.0,
        )
        assert limiter.decide("a").allowed
        assert limiter.decide("a").allowed
        assert not limiter.decide("a").allowed
        # Client b is untouched by a's exhausted bucket.
        decision = limiter.decide("b")
        assert decision.allowed
        assert decision.remaining == 1
        assert decision.limit == 2

    def test_denial_reports_retry_after(self) -> None:
        limiter = RateLimiter(
            RateLimitSettings(enabled=True, requests_per_minute=60, burst=1),
            clock=lambda: 0.0,
        )
        assert limiter.decide("a").allowed
        denied = limiter.decide("a")
        assert not denied.allowed
        # 60 rpm = 1 token/s; retry within ~1s.
        assert 0.0 < denied.retry_after <= 1.0
        assert denied.remaining == 0

    def test_clock_advance_refills(self) -> None:
        now = [0.0]

        def clock() -> float:
            return now[0]

        limiter = RateLimiter(
            RateLimitSettings(enabled=True, requests_per_minute=60, burst=1),
            clock=clock,
        )
        assert limiter.decide("a").allowed
        assert not limiter.decide("a").allowed
        now[0] = 2.0
        assert limiter.decide("a").allowed

    def test_bucket_table_is_bounded(self) -> None:
        limiter = RateLimiter(
            RateLimitSettings(enabled=True, requests_per_minute=60, burst=1),
            clock=lambda: 0.0,
        )
        for i in range(MAX_BUCKETS + 10):
            limiter.decide(f"key-{i}")
        assert len(limiter._buckets) == MAX_BUCKETS
        # The very first key was evicted; its counter restarted fresh.
        assert limiter.decide("key-0").remaining == 0

    def test_build_rate_limiter_disabled_returns_none(self) -> None:
        assert build_rate_limiter(
            Settings(rate_limit=RateLimitSettings(enabled=False))
        ) is None
        assert build_rate_limiter(Settings()) is not None


# ---------------------------------------------------------------------------
# Unit: client key derivation
# ---------------------------------------------------------------------------


class FakeHeaders:
    def __init__(self, data: dict[str, str]) -> None:
        self._data = {k.lower(): v for k, v in data.items()}

    def get(self, name: str, default=None):
        return self._data.get(name.lower(), default)


class TestClientKey:
    def test_credential_hash_stable_and_opaque(self) -> None:
        key1 = client_key(FakeHeaders({"Authorization": "Bearer sk-secret"}), None)
        key2 = client_key(FakeHeaders({"authorization": "Bearer sk-secret"}), "1.2.3.4")
        key3 = client_key(FakeHeaders({"Authorization": "Bearer other"}), None)
        assert key1 == key2  # same credential → same bucket regardless of IP
        assert key1 != key3
        assert "sk-secret" not in key1
        assert key1.startswith("cred:")

    def test_api_key_header_variants(self) -> None:
        assert client_key(FakeHeaders({"X-Api-Key": "k"}), None) == client_key(
            FakeHeaders({"x-api-key": "k"}), "9.9.9.9"
        )
        assert client_key(FakeHeaders({"api-key": "k"}), None).startswith("cred:")

    def test_falls_back_to_ip(self) -> None:
        assert client_key(FakeHeaders({}), "10.0.0.1") == "ip:10.0.0.1"
        assert client_key(FakeHeaders({}), None) == "ip:unknown"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    def test_defaults(self) -> None:
        settings = load_settings(environ={})
        assert settings.rate_limit.enabled is True
        assert settings.rate_limit.requests_per_minute == 60
        assert settings.rate_limit.burst == 20
        assert settings.max_body_bytes == 1_048_576

    def test_yaml_section(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text(
            "rate_limit:\n  enabled: false\n  requests_per_minute: 120\n"
            "  burst: 30\nmax_body_bytes: 2048\n",
            encoding="utf-8",
        )
        settings = load_settings(config_path=str(config_file), environ={})
        assert settings.rate_limit == RateLimitSettings(
            enabled=False, requests_per_minute=120, burst=30
        )
        assert settings.max_body_bytes == 2048

    def test_env_overrides_yaml(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text(
            "rate_limit:\n  requests_per_minute: 120\n  burst: 30\n",
            encoding="utf-8",
        )
        settings = load_settings(
            config_path=str(config_file),
            environ={
                "LMPI_RATE_LIMIT_ENABLED": "false",
                "LMPI_RATE_LIMIT_RPM": "10",
                "LMPI_RATE_LIMIT_BURST": "5",
                "LMPI_MAX_BODY_BYTES": "4096",
            },
        )
        assert settings.rate_limit.enabled is False
        assert settings.rate_limit.requests_per_minute == 10
        assert settings.rate_limit.burst == 5
        assert settings.max_body_bytes == 4096

    def test_invalid_values_raise(self) -> None:
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_RATE_LIMIT_RPM": "zero"})
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_RATE_LIMIT_RPM": "0"})
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_RATE_LIMIT_BURST": "-1"})
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_RATE_LIMIT_ENABLED": "maybe"})
        with pytest.raises(ValueError):
            load_settings(environ={"LMPI_MAX_BODY_BYTES": "0"})

    def test_non_mapping_section_raises(self, tmp_path) -> None:
        config_file = tmp_path / "lmpi.yaml"
        config_file.write_text("rate_limit: 42\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_settings(config_path=str(config_file), environ={})


# ---------------------------------------------------------------------------
# Integration: 429 / 413 / headers through the proxy
# ---------------------------------------------------------------------------


def post(client: TestClient, payload=COMPLETION_PAYLOAD, headers=None, content=None):
    body = content if content is not None else json.dumps(payload).encode("utf-8")
    return client.post(
        "/v1/chat/completions",
        content=body,
        headers={"content-type": "application/json", **(headers or {})},
    )


class TestRateLimitIntegration:
    def test_429_after_burst_with_retry_after(self) -> None:
        with make_client(
            ok_handler, rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=3)
        ) as client:
            for _ in range(3):
                response = post(client)
                assert response.status_code == 200
            limited = post(client)
            assert limited.status_code == 429
            assert limited.headers["retry-after"] == "1"
            assert limited.headers["x-ratelimit-limit"] == "3"
            assert limited.headers["x-ratelimit-remaining"] == "0"
            error = limited.json()["error"]
            assert error["type"] == "lmpi_rate_limited"
            assert error["code"] == 429

    def test_rate_headers_on_allowed_responses(self) -> None:
        with make_client(
            ok_handler, rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=5)
        ) as client:
            first = post(client)
            assert first.status_code == 200
            assert first.headers["x-ratelimit-limit"] == "5"
            assert first.headers["x-ratelimit-remaining"] == "4"
            second = post(client)
            assert second.headers["x-ratelimit-remaining"] == "3"

    def test_per_credential_buckets_are_independent(self) -> None:
        with make_client(
            ok_handler, rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=2)
        ) as client:
            for _ in range(2):
                assert post(client, headers={"Authorization": "Bearer a"}).status_code == 200
            assert (
                post(client, headers={"Authorization": "Bearer a"}).status_code == 429
            )
            # Different credential → different bucket → still allowed.
            assert post(client, headers={"Authorization": "Bearer b"}).status_code == 200
            # No credential → IP bucket, also independent.
            assert post(client).status_code == 200

    def test_rate_limit_applies_per_ip_when_no_credentials(self) -> None:
        with make_client(
            ok_handler, rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=1)
        ) as client:
            assert post(client).status_code == 200
            assert post(client).status_code == 429

    def test_sse_streaming_not_interrupted_by_rate_limit(self) -> None:
        chunks_yielded: list[bytes] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            async def stream() -> AsyncIterator[bytes]:
                yield b"data: {\"choices\": [{\"delta\": {\"content\": \"Hi\"}}]}\n\n"
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200,
                content=stream(),
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )

        with make_client(
            handler,
            rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=2),
        ) as client:
            payload = {**COMPLETION_PAYLOAD, "stream": True}
            with client.stream(
                "POST",
                "/v1/chat/completions",
                content=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["x-ratelimit-limit"] == "2"
                for chunk in response.iter_raw():
                    chunks_yielded.append(chunk)
        assert b"[DONE]" in b"".join(chunks_yielded)


class TestBodySizeLimitIntegration:
    def test_413_on_oversized_body(self) -> None:
        with make_client(ok_handler, max_body_bytes=64) as client:
            response = post(client, content=b"x" * 65)
            assert response.status_code == 413
            error = response.json()["error"]
            assert error["type"] == "lmpi_body_too_large"
            assert error["code"] == 413
            assert "64" in error["message"]

    def test_body_exactly_at_limit_passes(self) -> None:
        with make_client(ok_handler, max_body_bytes=64) as client:
            response = post(client, content=b"x" * 64)
            assert response.status_code == 200

    def test_normal_requests_unaffected_with_default_limit(self) -> None:
        with make_client(ok_handler) as client:
            response = post(client)
            assert response.status_code == 200

    def test_body_too_large_exception_carries_limit(self) -> None:
        with pytest.raises(BodyTooLarge) as exc_info:
            raise BodyTooLarge(1024)
        assert exc_info.value.limit == 1024


class TestHealthUnthrottled:
    def test_health_bypasses_rate_limit(self) -> None:
        with make_client(
            ok_handler, rate_limit=RateLimitSettings(enabled=True, requests_per_minute=60, burst=1)
        ) as client:
            # Drain the chat-completions bucket.
            assert post(client).status_code == 200
            assert post(client).status_code == 429
            # /health has no bucket: still 200 beyond the burst, every time.
            for _ in range(10):
                response = client.get("/health")
                assert response.status_code == 200
                assert "x-ratelimit-limit" not in response.headers
