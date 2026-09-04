"""Transparent proxying of chat completion requests to the upstream LLM API.

Handles non-streaming and SSE streaming (``"stream": true``) passthrough and
runs every parsed request through the detection pipeline hook before
forwarding. Responses are forwarded byte-for-byte; streaming responses are
pulled chunk-by-chunk from upstream and never buffered.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .detection.pipeline import DetectionPipeline, PipelineResult

logger = logging.getLogger("lmpi.proxy")

BLOCKED_STATUS_CODE = 403
BAD_GATEWAY_STATUS_CODE = 502

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


async def stream_upstream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield upstream response chunks as they arrive, without buffering."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.HTTPError as exc:
        # Mid-stream failure: emit an SSE error frame so the client can tell
        # the stream did not complete normally.
        logger.warning("Upstream stream aborted: %s", exc)
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


def blocked_response(reason: str | None) -> JSONResponse:
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
    )


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
    """POST /v1/chat/completions — detection hook + transparent upstream proxy."""
    client: httpx.AsyncClient = request.app.state.client
    pipeline: DetectionPipeline = request.app.state.pipeline

    body = await request.body()
    payload = parse_json_body(body)

    if payload is not None:
        result: PipelineResult = await pipeline.process_request(payload)
        if result.action == "block":
            logger.info("Request blocked by detection pipeline: %s", result.reason)
            return blocked_response(result.reason)
        if result.payload is not None:
            body = json.dumps(result.payload, ensure_ascii=False).encode("utf-8")

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
                    headers=filter_response_headers(upstream_response.headers),
                )
            return StreamingResponse(
                stream_upstream_response(upstream_response),
                status_code=upstream_response.status_code,
                headers=filter_response_headers(
                    upstream_response.headers, keep_content_encoding=True
                ),
            )
        upstream_response = await client.post(
            upstream_path, content=body, headers=headers
        )
    except httpx.HTTPError as exc:
        logger.warning("Upstream request failed: %s", exc)
        return bad_gateway_response(f"Upstream request failed: {type(exc).__name__}")

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=filter_response_headers(upstream_response.headers),
    )
