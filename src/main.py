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
from .canary import CanaryManager
from .config import Settings, load_settings
from .detection.pipeline import DetectionPipeline
from .deep_path import DeepPathDetector
from .deep_path.backend import OnnxRuntimeBackend
from .proxy import check_upstream, forward_chat_completions

logger = logging.getLogger("lmpi")


def build_deep_path_detector(settings: Settings) -> DeepPathDetector | None:
    """Build the stage-3 detector from settings; ``None`` = stage disabled.

    Graceful degradation (PLAN.md §3.1): when the stage is disabled, the
    model files are missing, or onnxruntime is not installed, this returns
    ``None`` and the pipeline simply skips the stage.
    """
    deep_path = settings.deep_path
    if not deep_path.enabled:
        return None
    try:
        backend = OnnxRuntimeBackend(deep_path.model_path)
    except Exception as exc:  # noqa: BLE001 - degrade instead of crash
        logger.warning(
            "Deep path stage disabled (backend unavailable): %s", exc
        )
        return None
    detector = DeepPathDetector(
        backend,
        block_threshold=deep_path.block_threshold,
        warn_threshold=deep_path.warn_threshold,
        max_chars=deep_path.max_chars,
    )
    if detector.available:
        logger.info(
            "Deep path enabled: model=%s quantized=%s block=%s warn=%s",
            backend.model_name,
            backend.quantized,
            deep_path.block_threshold,
            deep_path.warn_threshold,
        )
    return detector


def build_pipeline(settings: Settings) -> DetectionPipeline:
    """Build the detection pipeline with the configured stages."""
    return DetectionPipeline(
        normalization=settings.normalization,
        fast_path=settings.fast_path,
        deep_path_detector=build_deep_path_detector(settings),
    )


def build_canary_manager(settings: Settings) -> CanaryManager:
    """Build the canary manager (system prompt injection + leak scanning).

    Without ``LMPI_CANARY_SECRET`` an ephemeral secret is generated here
    (startup) with a warning — tokens then differ across restarts.
    """
    return CanaryManager(settings.canary)


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
    app.state.pipeline = build_pipeline(app.state.settings)
    app.state.canary = build_canary_manager(app.state.settings)

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
