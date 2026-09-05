"""Per-client rate limiting — in-memory token bucket (roadmap v1.1).

One token = one admitted request. Buckets refill continuously at
``requests_per_minute / 60`` tokens per second, capped at ``burst`` tokens,
so short spikes above the average rate are absorbed without allowing a
sustained rate above ``requests_per_minute``.

Limitations (documented in the README): the buckets live in a single
process — no cross-worker/Redis sharing (that is roadmap v3.5). The rate
limit applies to request *admission* only, never to an in-flight SSE
stream, so streaming responses are never interrupted mid-flight.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from fastapi import Request

from .config import RateLimitSettings, Settings

logger = logging.getLogger("lmpi.rate_limit")

RATE_LIMITED_STATUS_CODE = 429
BODY_TOO_LARGE_STATUS_CODE = 413

HEADER_RATE_LIMIT = "X-RateLimit-Limit"
HEADER_RATE_LIMIT_REMAINING = "X-RateLimit-Remaining"
HEADER_RETRY_AFTER = "Retry-After"

# Credentials in these headers (first present wins) identify the client more
# reliably than the source IP, which is shared behind proxies/NAT.
_CREDENTIAL_HEADERS = ("authorization", "x-api-key", "api-key")

# Hard cap on tracked buckets so unauthenticated/scanner traffic with
# unlimited distinct credentials cannot grow memory without bound.
MAX_BUCKETS = 65536


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of one admission check against a client's bucket."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: float  # seconds to wait; 0.0 when allowed


class TokenBucket:
    """Classic token bucket; capacity holds the burst allowance."""

    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        now: float | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be > 0, got {refill_per_second}")
        self.capacity = capacity
        self.refill_per_second = float(refill_per_second)
        # A fresh bucket starts full so the first burst of requests succeeds.
        self.tokens = float(capacity)
        self.updated = time.monotonic() if now is None else now

    def try_acquire(self, now: float | None = None) -> tuple[bool, float]:
        """Try to take one token; return ``(allowed, retry_after_seconds)``."""
        timestamp = time.monotonic() if now is None else now
        elapsed = max(0.0, timestamp - self.updated)
        self.tokens = min(
            float(self.capacity), self.tokens + elapsed * self.refill_per_second
        )
        self.updated = timestamp
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / self.refill_per_second

    def remaining(self) -> int:
        """Whole tokens currently available (0 when the bucket is drained)."""
        return int(self.tokens)


class _KeyRequest(Protocol):
    """Minimal request surface :func:`client_key` needs (also test fakes)."""

    def get(self, name: str, default: str | None = None) -> str | None: ...


def client_key(
    headers: _KeyRequest, client_host: str | None
) -> str:
    """Derive a stable per-client bucket key from a request.

    Hash of the credential (Authorization / API-key header) when present —
    hashed, never stored raw — otherwise the client IP. Returns
    ``"ip:unknown"`` when the ASGI server provides no client address
    (e.g. some test transports).
    """
    for header in _CREDENTIAL_HEADERS:
        value = headers.get(header)
        if value:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
            return f"cred:{digest}"
    return f"ip:{client_host or 'unknown'}"


def request_client_key(request: Request) -> str:
    """:func:`client_key` for a live FastAPI request."""
    return client_key(request.headers, request.client.host if request.client else None)


class RateLimiter:
    """Per-key token buckets, created lazily and kept in memory.

    Thread-safe (the ASGI event loop is single-threaded, but the limiter is
    also safe to touch from worker threads). When the bucket table exceeds
    :data:`MAX_BUCKETS` the oldest bucket is evicted — bounded memory at the
    cost of a rate-limit reset for extremely many simultaneous clients.
    """

    def __init__(
        self,
        settings: RateLimitSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._refill_per_second = settings.requests_per_minute / 60.0
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def decide(self, key: str) -> RateLimitDecision:
        """Run one admission check for ``key`` and consume a token if allowed."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= MAX_BUCKETS:
                    oldest = next(iter(self._buckets))
                    del self._buckets[oldest]
                bucket = TokenBucket(
                    self.settings.burst, self._refill_per_second, now=self._clock()
                )
                self._buckets[key] = bucket
            allowed, retry_after = bucket.try_acquire(now=self._clock())
            return RateLimitDecision(
                allowed=allowed,
                limit=self.settings.burst,
                remaining=bucket.remaining(),
                retry_after=0.0 if allowed else retry_after,
            )


def build_rate_limiter(settings: Settings) -> RateLimiter | None:
    """Build the shared rate limiter; ``None`` disables admission control."""
    if not settings.rate_limit.enabled:
        return None
    return RateLimiter(settings.rate_limit)
