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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger("lmpi.config")

DEFAULT_UPSTREAM_URL = "https://api.openai.com"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_NORMALIZATION_MODE = "rewrite"

ENV_PREFIX = "LMPI_"

# normalization.mode: what to do when the normalization stage finds something.
NORMALIZATION_MODES = ("rewrite", "block", "log")

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class NormalizationSettings:
    """Ingress normalization stage settings.

    ``mode`` controls the action when the stage finds something:
    ``rewrite`` (default) rewrites the payload with the cleaned text,
    ``block`` rejects the request with HTTP 403, ``log`` passes the request
    unchanged and only logs the findings.
    """

    mode: str = DEFAULT_NORMALIZATION_MODE
    unicode: bool = True
    base64: bool = True
    hex: bool = True
    rot13: bool = True
    delimiters: bool = True


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the LMPI proxy."""

    upstream_url: str = DEFAULT_UPSTREAM_URL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    config_path: str | None = None
    normalization: NormalizationSettings = field(
        default_factory=NormalizationSettings
    )


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


def _parse_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"normalization mode must be one of {', '.join(NORMALIZATION_MODES)}, "
            f"got {value!r}"
        )
    return mode


def _parse_bool(value: Any, env_key: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean for {env_key}: {value!r}")


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

    normalization_section = file_values.get("normalization")
    if normalization_section is None:
        normalization_section = {}
    if not isinstance(normalization_section, dict):
        raise ValueError(
            "normalization config section must be a mapping, got "
            f"{type(normalization_section).__name__}"
        )

    def norm_pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}NORMALIZATION_{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if normalization_section.get(key) is not None:
            return normalization_section[key]
        return default

    normalization = NormalizationSettings(
        mode=_parse_mode(norm_pick("mode", DEFAULT_NORMALIZATION_MODE)),
        unicode=_parse_bool(
            norm_pick("unicode", True), f"{ENV_PREFIX}NORMALIZATION_UNICODE"
        ),
        base64=_parse_bool(
            norm_pick("base64", True), f"{ENV_PREFIX}NORMALIZATION_BASE64"
        ),
        hex=_parse_bool(norm_pick("hex", True), f"{ENV_PREFIX}NORMALIZATION_HEX"),
        rot13=_parse_bool(norm_pick("rot13", True), f"{ENV_PREFIX}NORMALIZATION_ROT13"),
        delimiters=_parse_bool(
            norm_pick("delimiters", True),
            f"{ENV_PREFIX}NORMALIZATION_DELIMITERS",
        ),
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
        config_path=resolved_path,
        normalization=normalization,
    )
    logger.debug(
        "Loaded settings: upstream=%s host=%s port=%s timeout=%s "
        "normalization.mode=%s",
        settings.upstream_url,
        settings.host,
        settings.port,
        settings.request_timeout,
        settings.normalization.mode,
    )
    return settings
