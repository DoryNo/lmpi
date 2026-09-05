"""Transparent proxying of chat completion requests to the upstream LLM API.

Handles non-streaming and SSE streaming (``"stream": true``) passthrough and
runs every parsed request through the detection pipeline hook before
forwarding. Responses are forwarded byte-for-byte; streaming responses are
pulled chunk-by-chunk from upstream and never buffered.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, AsyncIterator

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .canary import CanaryManager, CanaryScanState, CanaryToken
from .detection.pipeline import DetectionPipeline, PipelineResult
from .rate_limit import (
    BODY_TOO_LARGE_STATUS_CODE,
    HEADER_RATE_LIMIT,
    HEADER_RATE_LIMIT_REMAINING,
    HEADER_RETRY_AFTER,
    RATE_LIMITED_STATUS_CODE,
    RateLimitDecision,
    request_client_key,
)

logger = logging.getLogger("lmpi.proxy")

BLOCKED_STATUS_CODE = 403
BAD_GATEWAY_STATUS_CODE = 502

RATE_LIMITED_MESSAGE = "LMPI rate limit exceeded: too many requests"


class BodyTooLarge(Exception):
    """Raised when the request body exceeds ``max_body_bytes``."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"request body exceeds {limit} bytes")
        self.limit = limit

# Hop-by-hop headers (RFC 9110 section 7.6.1) plus headers httpx recomputes
# itself for the upstream request (host from the upstream URL, content-length
# from the actual body).
_REQUEST_STRIP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Response headers we never pass through: hop-by-hop, plus "date"/"server"
# (the ASGI server adds its own) and "content-length" (Starlette recomputes it;
# for streaming there is no fixed length).
_RESPONSE_STRIP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "date",
        "server",
        "content-length",
    }
)


def filter_request_headers(headers: Any) -> dict[str, str]:
    """Return upstream-bound headers: end-to-end headers kept, hop-by-hop dropped."""
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _REQUEST_STRIP_HEADERS
    }


def filter_response_headers(
    headers: Any, *, keep_content_encoding: bool = False
) -> dict[str, str]:
    """Return client-bound headers copied from the upstream response.

    ``keep_content_encoding`` is set when the body is forwarded raw
    (streaming path); otherwise httpx already decoded the content, so the
    ``content-encoding`` header must be dropped to match.
    """
    stripped = _RESPONSE_STRIP_HEADERS
    if not keep_content_encoding:
        stripped = stripped | {"content-encoding"}
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in stripped
    }


def parse_json_body(body: bytes) -> dict[str, Any] | None:
    """Parse the request body as a JSON object; ``None`` for anything else."""
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def sse_error_event(exc: Exception) -> bytes:
    """Render an SSE ``event: error`` frame for mid-stream upstream failures."""
    message = f"LMPI upstream error: {type(exc).__name__}"
    frame = {"error": {"message": message, "type": "lmpi_bad_gateway"}}
    return f"event: error\ndata: {json.dumps(frame)}\n\n".encode("utf-8")


LEAK_DETECTED_MESSAGE = (
    "LMPI leak detected: canary token found in upstream response"
)


def sse_leak_event() -> bytes:
    """Render an SSE ``event: error`` frame terminating a stream on a leak."""
    frame = {"error": {"message": LEAK_DETECTED_MESSAGE, "type": "lmpi_leak_detected"}}
    return f"event: error\ndata: {json.dumps(frame)}\n\n".encode("utf-8")


async def stream_upstream_response(
    response: httpx.Response, scan: CanaryScanState | None = None
) -> AsyncIterator[bytes]:
    """Yield upstream response chunks as they arrive, without buffering.

    When ``scan`` is set, chunks pass through the request-scoped canary
    scanner first (redaction / block termination) instead of going straight
    to the client.
    """
    try:
        async for chunk in response.aiter_raw():
            if scan is None:
                yield chunk
                continue
            emitted = scan.process(chunk)
            if emitted:
                yield emitted
            if scan.action == "block" and scan.leaked:
                yield sse_leak_event()
                return
        if scan is not None:
            tail = scan.flush()
            if tail:
                yield tail
    except httpx.HTTPError as exc:
        # Mid-stream failure: emit an SSE error frame so the client can tell
        # the stream did not complete normally. Flush the held-back tail
        # first so no already-scanned content is lost.
        logger.warning("Upstream stream aborted: %s", exc)
        if scan is not None:
            tail = scan.flush()
            if tail:
                yield tail
        yield sse_error_event(exc)
    finally:
        await response.aclose()


def bad_gateway_response(detail: str) -> JSONResponse:
    """502 response for upstream connection/request failures."""
    return JSONResponse(
        status_code=BAD_GATEWAY_STATUS_CODE,
        content={
            "error": {
                "message": detail,
                "type": "lmpi_bad_gateway",
                "code": BAD_GATEWAY_STATUS_CODE,
            }
        },
    )


def blocked_response(
    reason: str | None, extra_headers: dict[str, str] | None = None
) -> JSONResponse:
    """403 response used when the detection pipeline blocks a request."""
    return JSONResponse(
        status_code=BLOCKED_STATUS_CODE,
        content={
            "error": {
                "message": reason or "Request blocked by LMPI",
                "type": "lmpi_policy_block",
                "code": BLOCKED_STATUS_CODE,
            }
        },
        headers=extra_headers or {},
    )


def leak_detected_response() -> JSONResponse:
    """502 response used when a canary token leaks into the response body."""
    return JSONResponse(
        status_code=BAD_GATEWAY_STATUS_CODE,
        content={
            "error": {
                "message": LEAK_DETECTED_MESSAGE,
                "type": "lmpi_leak_detected",
                "code": BAD_GATEWAY_STATUS_CODE,
            }
        },
    )


def rate_limit_headers(decision: RateLimitDecision | None) -> dict[str, str]:
    """X-RateLimit-* headers to attach when rate limiting is enabled."""
    if decision is None:
        return {}
    return {
        HEADER_RATE_LIMIT: str(decision.limit),
        HEADER_RATE_LIMIT_REMAINING: str(decision.remaining),
    }


def rate_limited_response(decision: RateLimitDecision) -> JSONResponse:
    """429 response with Retry-After after the client's bucket ran dry."""
    retry_after = max(1, int(math.ceil(decision.retry_after)))
    return JSONResponse(
        status_code=RATE_LIMITED_STATUS_CODE,
        content={
            "error": {
                "message": RATE_LIMITED_MESSAGE,
                "type": "lmpi_rate_limited",
                "code": RATE_LIMITED_STATUS_CODE,
            }
        },
        headers={
            **rate_limit_headers(decision),
            HEADER_RETRY_AFTER: str(retry_after),
        },
    )


def body_too_large_response(
    limit: int, extra_headers: dict[str, str] | None = None
) -> JSONResponse:
    """413 response for bodies exceeding ``max_body_bytes`` (DoS hardening)."""
    return JSONResponse(
        status_code=BODY_TOO_LARGE_STATUS_CODE,
        content={
            "error": {
                "message": f"Request body exceeds the {limit} byte limit",
                "type": "lmpi_body_too_large",
                "code": BODY_TOO_LARGE_STATUS_CODE,
            }
        },
        headers=extra_headers or {},
    )


async def read_body_limited(request: Request, max_bytes: int) -> bytes:
    """Read the request body while enforcing the size cap.

    Streams chunk-by-chunk and aborts as soon as the cap is exceeded, so an
    oversized body is never fully buffered. Raises :class:`BodyTooLarge`.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise BodyTooLarge(max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


async def check_upstream(
    client: httpx.AsyncClient, upstream_url: str
) -> dict[str, Any]:
    """Probe upstream reachability for the /health endpoint."""
    try:
        response = await client.get(upstream_url, timeout=5.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.warning("Upstream health probe failed: %s", exc)
        return {"url": upstream_url, "reachable": False, "error": type(exc).__name__}
    return {
        "url": upstream_url,
        "reachable": True,
        "status_code": response.status_code,
    }


async def forward_chat_completions(request: Request) -> Response:
    """POST /v1/chat/completions — detection hook + transparent upstream proxy.

    Canary injection runs AFTER the detection pipeline rewrite, on the final
    payload dict, so the canary lands in exactly the system prompt the
    upstream LLM will see. The response (body or stream) is then scanned for
    that same per-request token.
    """
    client: httpx.AsyncClient = request.app.state.client
    pipeline: DetectionPipeline = request.app.state.pipeline
    canary: CanaryManager | None = getattr(request.app.state, "canary", None)
    settings = request.app.state.settings

    # --- Admission control: rate limit + body-size cap (v1.1 hardening) ---
    limiter = getattr(request.app.state, "rate_limiter", None)
    decision: RateLimitDecision | None = None
    if limiter is not None:
        decision = limiter.decide(request_client_key(request))
        if not decision.allowed:
            logger.info(
                "Rate limit exceeded (remaining=0, retry_after=%.2fs)",
                decision.retry_after,
            )
            return rate_limited_response(decision)
    extra_headers = rate_limit_headers(decision)

    try:
        body = await read_body_limited(request, settings.max_body_bytes)
    except BodyTooLarge:
        logger.warning(
            "Request body exceeds max_body_bytes=%s — rejected with 413",
            settings.max_body_bytes,
        )
        return body_too_large_response(settings.max_body_bytes, extra_headers)

    payload = parse_json_body(body)
    canary_token: CanaryToken | None = None

    if payload is not None:
        result: PipelineResult = await pipeline.process_request(payload)
        if result.action == "block":
            logger.info("Request blocked by detection pipeline: %s", result.reason)
            return blocked_response(result.reason, extra_headers)
        final_payload = payload if result.payload is None else result.payload
        if canary is not None:
            final_payload, canary_token = canary.inject(final_payload)
        if result.payload is not None or canary_token is not None:
            body = json.dumps(final_payload, ensure_ascii=False).encode("utf-8")

    upstream_path = request.url.path
    if request.url.query:
        upstream_path = f"{upstream_path}?{request.url.query}"
    headers = filter_request_headers(request.headers)
    stream = payload is not None and bool(payload.get("stream"))

    try:
        if stream:
            upstream_request = client.build_request(
                "POST", upstream_path, content=body, headers=headers
            )
            upstream_response = await client.send(upstream_request, stream=True)
            if upstream_response.status_code >= 400:
                # Error bodies are small JSON — drain and return them as a
                # regular (non-streamed) response with the upstream status.
                content = await upstream_response.aread()
                await upstream_response.aclose()
                return Response(
                    content=content,
                    status_code=upstream_response.status_code,
                    headers={
                        **filter_response_headers(upstream_response.headers),
                        **extra_headers,
                    },
                )
            scan = (
                canary.new_scan_state(canary_token)
                if canary is not None and canary_token is not None
                else None
            )
            return StreamingResponse(
                stream_upstream_response(upstream_response, scan=scan),
                status_code=upstream_response.status_code,
                headers={
                    **filter_response_headers(
                        upstream_response.headers, keep_content_encoding=True
                    ),
                    **extra_headers,
                },
            )
        upstream_response = await client.post(
            upstream_path, content=body, headers=headers
        )
    except httpx.HTTPError as exc:
        logger.warning("Upstream request failed: %s", exc)
        return bad_gateway_response(f"Upstream request failed: {type(exc).__name__}")

    content = upstream_response.content
    if canary is not None and canary_token is not None:
        content, scan = canary.scan_bytes(content, canary_token)
        if scan.action == "block" and scan.leaked:
            return leak_detected_response()

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers={
            **filter_response_headers(upstream_response.headers),
            **extra_headers,
        },
    )
