"""Configuration loading for LMPI.

Precedence (highest first):

1. Environment variables (``LMPI_*``)
2. YAML file (path from ``LMPI_CONFIG_PATH``, or ``config.yaml`` keys)
3. Built-in defaults

Unknown YAML keys are ignored so future agents (detection stages, benchmarks)
can add their own sections without breaking the proxy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fast_path.detector import (
    DEFAULT_BLOCK_THRESHOLD as DEFAULT_FAST_PATH_BLOCK_THRESHOLD,
    DEFAULT_WARN_THRESHOLD as DEFAULT_FAST_PATH_WARN_THRESHOLD,
)

logger = logging.getLogger("lmpi.config")

DEFAULT_UPSTREAM_URL = "https://api.openai.com"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_REQUEST_TIMEOUT = 300.0

ENV_PREFIX = "LMPI_"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the LMPI proxy.

    Fast-path settings (Agent 3):

    - ``fast_path_enabled`` — run the regex/heuristic stage in the pipeline
    - ``fast_path_block_threshold`` — score at/above which requests are
      blocked with HTTP 403
    - ``fast_path_warn_threshold`` — score at/above which requests are only
      logged (warn) but still forwarded
    """

    upstream_url: str = DEFAULT_UPSTREAM_URL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    fast_path_enabled: bool = True
    fast_path_block_threshold: float = DEFAULT_FAST_PATH_BLOCK_THRESHOLD
    fast_path_warn_threshold: float = DEFAULT_FAST_PATH_WARN_THRESHOLD
    config_path: str | None = None


def _parse_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid port value: {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Port out of range (1-65535): {port}")
    return port


def _parse_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid request_timeout value: {value!r}") from exc
    if timeout <= 0:
        raise ValueError(f"request_timeout must be positive, got {timeout}")
    return timeout


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_threshold(value: Any, *, name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} value: {value!r}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {threshold}")
    return threshold


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LMPI config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"LMPI config file must contain a mapping, got {type(data).__name__}: {path}"
        )
    return data


def load_settings(
    config_path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Build :class:`Settings` from YAML file + environment variables.

    Env vars always win over YAML values. ``environ`` is injectable so tests
    never depend on the real process environment.
    """
    env = os.environ if environ is None else environ

    resolved_path = config_path or env.get(f"{ENV_PREFIX}CONFIG_PATH")
    file_values: dict[str, Any] = {}
    if resolved_path:
        file_values = _load_yaml(Path(resolved_path))

    def pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if file_values.get(key) is not None:
            return file_values[key]
        return default

    upstream_url = str(pick("upstream_url", DEFAULT_UPSTREAM_URL)).rstrip("/")
    host = str(pick("host", DEFAULT_HOST))
    port = _parse_port(pick("port", DEFAULT_PORT))
    request_timeout = _parse_timeout(pick("request_timeout", DEFAULT_REQUEST_TIMEOUT))

    fast_path_enabled = _parse_bool(pick("fast_path_enabled", True))
    fast_path_block_threshold = _parse_threshold(
        pick("fast_path_block_threshold", DEFAULT_FAST_PATH_BLOCK_THRESHOLD),
        name="fast_path_block_threshold",
    )
    fast_path_warn_threshold = _parse_threshold(
        pick("fast_path_warn_threshold", DEFAULT_FAST_PATH_WARN_THRESHOLD),
        name="fast_path_warn_threshold",
    )
    if fast_path_warn_threshold > fast_path_block_threshold:
        raise ValueError(
            f"fast_path_warn_threshold ({fast_path_warn_threshold}) must not "
            f"exceed fast_path_block_threshold ({fast_path_block_threshold})"
        )

    if not upstream_url.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"upstream_url must start with http:// or https://, got {upstream_url!r}"
        )

    settings = Settings(
        upstream_url=upstream_url,
        host=host,
        port=port,
        request_timeout=request_timeout,
        fast_path_enabled=fast_path_enabled,
        fast_path_block_threshold=fast_path_block_threshold,
        fast_path_warn_threshold=fast_path_warn_threshold,
        config_path=resolved_path,
    )
    logger.debug(
        "Loaded settings: upstream=%s host=%s port=%s timeout=%ss "
        "fast_path=%s block=%s warn=%s",
        settings.upstream_url,
        settings.host,
        settings.port,
        settings.request_timeout,
        settings.fast_path_enabled,
        settings.fast_path_block_threshold,
        settings.fast_path_warn_threshold,
    )
    return settings
