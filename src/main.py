"""LMPI FastAPI application entrypoint.

Run with::

    uvicorn src.main:app --host 0.0.0.0 --port 8080

or ``python -m src.main`` to pick host/port from ``LMPI_*`` env vars /
``config.yaml``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from . import __version__
from .config import Settings, load_settings
from .detection.pipeline import DetectionPipeline
from .proxy import check_upstream, forward_chat_completions

logger = logging.getLogger("lmpi")


def build_upstream_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """Create the httpx client used to talk to the upstream LLM API.

    ``transport`` is a testing hook (``httpx.MockTransport``) so tests never
    touch the network.
    """
    return httpx.AsyncClient(
        base_url=settings.upstream_url,
        timeout=httpx.Timeout(
            connect=10.0,
            read=settings.request_timeout,
            write=30.0,
            pool=10.0,
        ),
        transport=transport,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared upstream client on startup, close it on shutdown."""
    app.state.client = build_upstream_client(
        app.state.settings, transport=app.state.transport
    )
    logger.info(
        "LMPI proxy ready → upstream %s", app.state.settings.upstream_url
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the LMPI FastAPI app."""
    app = FastAPI(title="LMPI Proxy", version=__version__, lifespan=lifespan)
    app.state.settings = settings or load_settings()
    app.state.transport = transport
    app.state.pipeline = DetectionPipeline()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await forward_chat_completions(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        upstream = await check_upstream(
            app.state.client, app.state.settings.upstream_url
        )
        return {"status": "ok", "version": __version__, "upstream": upstream}

    return app


def main() -> None:
    """Run uvicorn with host/port from settings (env vars > YAML > defaults)."""
    settings = load_settings()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Starting LMPI proxy on %s:%s → upstream %s",
        settings.host,
        settings.port,
        settings.upstream_url,
    )
    uvicorn.run("src.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()


app = create_app()
