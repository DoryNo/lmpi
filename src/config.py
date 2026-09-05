"""Configuration loading for LMPI.

Precedence (highest first):

1. Environment variables (``LMPI_*``)
2. YAML file (path from ``LMPI_CONFIG_PATH``, or ``config.yaml`` keys)
3. Built-in defaults

Unknown YAML keys are reported with a warning (so typos fail loudly enough
to notice) but do not break the proxy; invalid *values* (types, ranges,
mutually inconsistent options) raise :class:`ValueError` with a message
naming the offending key and value so startup fails fast and clearly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fast_path.detector import (
    DEFAULT_BLOCK_THRESHOLD as DEFAULT_FAST_PATH_BLOCK_THRESHOLD,
    DEFAULT_WARN_THRESHOLD as DEFAULT_FAST_PATH_WARN_THRESHOLD,
)
from .deep_path.detector import (
    DEFAULT_BLOCK_THRESHOLD as DEFAULT_DEEP_PATH_BLOCK_THRESHOLD,
    DEFAULT_MAX_CHARS as DEFAULT_DEEP_PATH_MAX_CHARS,
    DEFAULT_WARN_THRESHOLD as DEFAULT_DEEP_PATH_WARN_THRESHOLD,
)

logger = logging.getLogger("lmpi.config")

DEFAULT_UPSTREAM_URL = "https://api.openai.com"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_NORMALIZATION_MODE = "rewrite"
DEFAULT_CANARY_ACTION = "redact"
DEFAULT_DEEP_PATH_MODEL_PATH = "models/deberta-v3-base-prompt-injection-v2"
DEFAULT_RATE_LIMIT_ENABLED = True
DEFAULT_RATE_LIMIT_RPM = 60
DEFAULT_RATE_LIMIT_BURST = 20
DEFAULT_MAX_BODY_BYTES = 1_048_576  # 1 MiB

ENV_PREFIX = "LMPI_"

# normalization.mode: what to do when the normalization stage finds something.
NORMALIZATION_MODES = ("rewrite", "block", "log")

# canary.action: what to do when a canary token leaks into a response.
CANARY_ACTIONS = ("redact", "block")

# audit.path: special targets that map to process streams; anything else is
# treated as a file path (opened in append mode).
AUDIT_STREAM_TARGETS = frozenset({"stdout", "stderr"})

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
class FastPathSettings:
    """Fast-path (stage 2) settings.

    ``enabled`` toggles the regex/heuristic stage; ``block_threshold`` is
    the composite noisy-OR score at/above which the request is rejected with
    HTTP 403; ``warn_threshold`` is the score at/above which the request is
    logged (warn) but still forwarded. Both thresholds are heuristics
    pending tuning against the benchmark eval set (Agent 7).
    """

    enabled: bool = True
    block_threshold: float = DEFAULT_FAST_PATH_BLOCK_THRESHOLD
    warn_threshold: float = DEFAULT_FAST_PATH_WARN_THRESHOLD


@dataclass(frozen=True)
class DeepPathSettings:
    """Deep-path (stage 3) settings — ONNX ML classifier.

    ``enabled`` defaults to **False** until the model has been downloaded
    with ``scripts/download_model.py`` (models/ is gitignored);
    ``model_path`` is the directory holding ``model.onnx`` (or
    ``model_quantized.onnx``) + ``tokenizer.json``; ``block_threshold`` /
    ``warn_threshold`` are injection-probability cutoffs (same semantics as
    the fast path); ``max_chars`` caps the text handed to the classifier so
    oversized prompts cannot burn inference time. Thresholds are heuristics
    pending tuning against the benchmark eval set (Agent 7).
    """

    enabled: bool = False
    model_path: str = DEFAULT_DEEP_PATH_MODEL_PATH
    block_threshold: float = DEFAULT_DEEP_PATH_BLOCK_THRESHOLD
    warn_threshold: float = DEFAULT_DEEP_PATH_WARN_THRESHOLD
    max_chars: int = DEFAULT_DEEP_PATH_MAX_CHARS


@dataclass(frozen=True)
class CanarySettings:
    """Canary token (system prompt leak detection) settings.

    ``enabled`` defaults to **True** with ``action="redact"`` — leak
    detection on by default, and a leaked canary is replaced with
    ``[REDACTED]`` rather than killing the response. When ``secret`` is
    unset the manager generates an ephemeral per-process secret and logs a
    warning (tokens are then not reproducible across restarts).
    ``add_missing_system`` opts in to injecting a system message when the
    request has none; by default the proxy stays transparent.
    """

    enabled: bool = True
    secret: str | None = None
    action: str = DEFAULT_CANARY_ACTION
    add_missing_system: bool = False


@dataclass(frozen=True)
class RateLimitSettings:
    """Per-client rate limiting (in-memory token bucket) settings.

    ``requests_per_minute`` is the sustained refill rate and ``burst`` the
    bucket capacity, so callers may briefly exceed the average rate. The
    limiter is keyed per client (credential hash, else IP) and lives in a
    single process — see :mod:`src.rate_limit`.
    """

    enabled: bool = DEFAULT_RATE_LIMIT_ENABLED
    requests_per_minute: int = DEFAULT_RATE_LIMIT_RPM
    burst: int = DEFAULT_RATE_LIMIT_BURST


@dataclass(frozen=True)
class AuditSettings:
    """Audit / access log settings (Agent 10, hardening v1.1).

    ``enabled`` (default **False**) turns the JSONL audit log on. ``path``
    selects the sink: ``stdout``, ``stderr``, or a file path (append mode,
    closed on graceful shutdown). ``include_text`` (default **False**)
    controls whether prompt text is embedded in detection events; when it
    is off the ``text`` field is omitted entirely, and when on, any canary
    token value is scrubbed by the caller before the event is written.
    ``access_log`` additionally emits one access-log entry per HTTP request
    (method, path, status, duration, hashed client key — never bodies or
    raw credentials).
    """

    enabled: bool = False
    path: str = "stdout"
    include_text: bool = False
    access_log: bool = False


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the LMPI proxy."""

    upstream_url: str = DEFAULT_UPSTREAM_URL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    config_path: str | None = None
    normalization: NormalizationSettings = field(
        default_factory=NormalizationSettings
    )
    fast_path: FastPathSettings = field(default_factory=FastPathSettings)
    deep_path: DeepPathSettings = field(default_factory=DeepPathSettings)
    canary: CanarySettings = field(default_factory=CanarySettings)
    rate_limit: RateLimitSettings = field(default_factory=RateLimitSettings)
    audit: AuditSettings = field(default_factory=AuditSettings)


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


def _parse_threshold(value: Any, *, name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} value: {value!r}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {threshold}")
    return threshold


def _parse_canary_action(value: Any) -> str:
    action = str(value).strip().lower()
    if action not in CANARY_ACTIONS:
        raise ValueError(
            f"canary action must be one of {', '.join(CANARY_ACTIONS)}, "
            f"got {value!r}"
        )
    return action


def _parse_canary_secret(value: Any) -> str | None:
    """Empty/whitespace secret values fall back to the ephemeral default."""
    if value is None:
        return None
    secret = str(value).strip()
    return secret or None


def _parse_positive_int(value: Any, *, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} value: {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {number}")
    return number


def _parse_audit_path(value: Any) -> str:
    path = str(value).strip()
    if not path:
        raise ValueError(
            "audit path must be 'stdout', 'stderr', or a file path, got empty value"
        )
    return path


# Top-level YAML keys accepted by load_settings; anything else triggers a
# warning (typo protection) without failing startup.
KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "upstream_url",
        "host",
        "port",
        "request_timeout",
        "max_body_bytes",
        "normalization",
        "fast_path",
        "deep_path",
        "canary",
        "rate_limit",
        "audit",
    }
)

# Accepted keys per YAML section (must stay in sync with the dataclasses).
KNOWN_SECTION_KEYS: dict[str, frozenset[str]] = {
    "normalization": frozenset(
        {"mode", "unicode", "base64", "hex", "rot13", "delimiters"}
    ),
    "fast_path": frozenset({"enabled", "block_threshold", "warn_threshold"}),
    "deep_path": frozenset(
        {"enabled", "model_path", "block_threshold", "warn_threshold", "max_chars"}
    ),
    "canary": frozenset({"enabled", "secret", "action", "add_missing_system"}),
    "rate_limit": frozenset({"enabled", "requests_per_minute", "burst"}),
    "audit": frozenset({"enabled", "path", "include_text", "access_log"}),
}


def warn_unknown_keys(
    file_values: dict[str, Any],
    *,
    logger: logging.Logger,
    section: str | None = None,
) -> None:
    """Log a warning for YAML keys we do not recognize (typo protection)."""
    if section is None:
        known = KNOWN_TOP_LEVEL_KEYS
        unknown = [key for key in file_values if key not in known]
    else:
        known = KNOWN_SECTION_KEYS.get(section, frozenset())
        unknown = [key for key in file_values if key not in known]
    if unknown:
        scope = f" in section '{section}'" if section else ""
        logger.warning(
            "Unknown config key(s)%s (ignored, check for typos): %s",
            scope,
            ", ".join(sorted(unknown)),
        )


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
        warn_unknown_keys(file_values, logger=logger)

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
    warn_unknown_keys(normalization_section, logger=logger, section="normalization")

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

    fast_path_section = file_values.get("fast_path")
    if fast_path_section is None:
        fast_path_section = {}
    if not isinstance(fast_path_section, dict):
        raise ValueError(
            "fast_path config section must be a mapping, got "
            f"{type(fast_path_section).__name__}"
        )
    warn_unknown_keys(fast_path_section, logger=logger, section="fast_path")

    def fp_pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}FAST_PATH_{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if fast_path_section.get(key) is not None:
            return fast_path_section[key]
        return default

    fast_path = FastPathSettings(
        enabled=_parse_bool(
            fp_pick("enabled", True), f"{ENV_PREFIX}FAST_PATH_ENABLED"
        ),
        block_threshold=_parse_threshold(
            fp_pick("block_threshold", DEFAULT_FAST_PATH_BLOCK_THRESHOLD),
            name="fast_path_block_threshold",
        ),
        warn_threshold=_parse_threshold(
            fp_pick("warn_threshold", DEFAULT_FAST_PATH_WARN_THRESHOLD),
            name="fast_path_warn_threshold",
        ),
    )
    if fast_path.warn_threshold > fast_path.block_threshold:
        raise ValueError(
            f"fast_path_warn_threshold ({fast_path.warn_threshold}) must not "
            f"exceed fast_path_block_threshold ({fast_path.block_threshold})"
        )

    # ---------------------------------------------------------------------
    # Deep path (stage 3) — ML classifier settings. Disabled by default
    # until the ONNX model has been downloaded (scripts/download_model.py).
    # ---------------------------------------------------------------------
    deep_path_section = file_values.get("deep_path")
    if deep_path_section is None:
        deep_path_section = {}
    if not isinstance(deep_path_section, dict):
        raise ValueError(
            "deep_path config section must be a mapping, got "
            f"{type(deep_path_section).__name__}"
        )
    warn_unknown_keys(deep_path_section, logger=logger, section="deep_path")

    def dp_pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}DEEP_PATH_{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if deep_path_section.get(key) is not None:
            return deep_path_section[key]
        return default

    deep_path = DeepPathSettings(
        enabled=_parse_bool(
            dp_pick("enabled", False), f"{ENV_PREFIX}DEEP_PATH_ENABLED"
        ),
        model_path=str(
            dp_pick("model_path", DEFAULT_DEEP_PATH_MODEL_PATH)
        ),
        block_threshold=_parse_threshold(
            dp_pick("block_threshold", DEFAULT_DEEP_PATH_BLOCK_THRESHOLD),
            name="deep_path_block_threshold",
        ),
        warn_threshold=_parse_threshold(
            dp_pick("warn_threshold", DEFAULT_DEEP_PATH_WARN_THRESHOLD),
            name="deep_path_warn_threshold",
        ),
        max_chars=_parse_positive_int(
            dp_pick("max_chars", DEFAULT_DEEP_PATH_MAX_CHARS),
            name="deep_path_max_chars",
        ),
    )
    if deep_path.warn_threshold > deep_path.block_threshold:
        raise ValueError(
            f"deep_path_warn_threshold ({deep_path.warn_threshold}) must not "
            f"exceed deep_path_block_threshold ({deep_path.block_threshold})"
        )

    canary_section = file_values.get("canary")
    if canary_section is None:
        canary_section = {}
    if not isinstance(canary_section, dict):
        raise ValueError(
            "canary config section must be a mapping, got "
            f"{type(canary_section).__name__}"
        )
    warn_unknown_keys(canary_section, logger=logger, section="canary")

    def canary_pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}CANARY_{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if canary_section.get(key) is not None:
            return canary_section[key]
        return default

    canary = CanarySettings(
        enabled=_parse_bool(
            canary_pick("enabled", True), f"{ENV_PREFIX}CANARY_ENABLED"
        ),
        secret=_parse_canary_secret(canary_pick("secret", None)),
        action=_parse_canary_action(canary_pick("action", DEFAULT_CANARY_ACTION)),
        add_missing_system=_parse_bool(
            canary_pick("add_missing_system", False),
            f"{ENV_PREFIX}CANARY_ADD_MISSING_SYSTEM",
        ),
    )

    # ---------------------------------------------------------------------
    # Rate limiting (v1.1 hardening) — per-client token bucket.
    # ---------------------------------------------------------------------
    rate_limit_section = file_values.get("rate_limit")
    if rate_limit_section is None:
        rate_limit_section = {}
    if not isinstance(rate_limit_section, dict):
        raise ValueError(
            "rate_limit config section must be a mapping, got "
            f"{type(rate_limit_section).__name__}"
        )
    warn_unknown_keys(rate_limit_section, logger=logger, section="rate_limit")

    # requests_per_minute is spelled LMPI_RATE_LIMIT_RPM for brevity.
    rate_limit_env_keys = {
        "enabled": f"{ENV_PREFIX}RATE_LIMIT_ENABLED",
        "requests_per_minute": f"{ENV_PREFIX}RATE_LIMIT_RPM",
        "burst": f"{ENV_PREFIX}RATE_LIMIT_BURST",
    }

    def rl_pick(key: str, default: Any) -> Any:
        env_key = rate_limit_env_keys[key]
        if env.get(env_key):
            return env[env_key]
        if rate_limit_section.get(key) is not None:
            return rate_limit_section[key]
        return default

    rate_limit = RateLimitSettings(
        enabled=_parse_bool(rl_pick("enabled", DEFAULT_RATE_LIMIT_ENABLED), rate_limit_env_keys["enabled"]),
        requests_per_minute=_parse_positive_int(
            rl_pick("requests_per_minute", DEFAULT_RATE_LIMIT_RPM),
            name="rate_limit_requests_per_minute",
        ),
        burst=_parse_positive_int(
            rl_pick("burst", DEFAULT_RATE_LIMIT_BURST),
            name="rate_limit_burst",
        ),
    )

    max_body_bytes = _parse_positive_int(
        pick("max_body_bytes", DEFAULT_MAX_BODY_BYTES),
        name="max_body_bytes",
    )

    # ---------------------------------------------------------------------
    # Audit / access log (Agent 10, hardening v1.1).
    # ---------------------------------------------------------------------
    rate_limit_section = file_values.get("rate_limit")
    if rate_limit_section is None:
        rate_limit_section = {}
    if not isinstance(rate_limit_section, dict):
        raise ValueError(
            "rate_limit config section must be a mapping, got "
            f"{type(rate_limit_section).__name__}"
        )

    # requests_per_minute is spelled LMPI_RATE_LIMIT_RPM for brevity.
    rate_limit_env_keys = {
        "enabled": f"{ENV_PREFIX}RATE_LIMIT_ENABLED",
        "requests_per_minute": f"{ENV_PREFIX}RATE_LIMIT_RPM",
        "burst": f"{ENV_PREFIX}RATE_LIMIT_BURST",
    }

    def rl_pick(key: str, default: Any) -> Any:
        env_key = rate_limit_env_keys[key]
        if env.get(env_key):
            return env[env_key]
        if rate_limit_section.get(key) is not None:
            return rate_limit_section[key]
        return default

    rate_limit = RateLimitSettings(
        enabled=_parse_bool(rl_pick("enabled", DEFAULT_RATE_LIMIT_ENABLED), rate_limit_env_keys["enabled"]),
        requests_per_minute=_parse_positive_int(
            rl_pick("requests_per_minute", DEFAULT_RATE_LIMIT_RPM),
            name="rate_limit_requests_per_minute",
        ),
        burst=_parse_positive_int(
            rl_pick("burst", DEFAULT_RATE_LIMIT_BURST),
            name="rate_limit_burst",
        ),
    )

    max_body_bytes = _parse_positive_int(
        pick("max_body_bytes", DEFAULT_MAX_BODY_BYTES),
        name="max_body_bytes",
    )

    # ---------------------------------------------------------------------
    # Audit / access log (Agent 10, hardening v1.1).
    # ---------------------------------------------------------------------
    audit_section = file_values.get("audit")
    if audit_section is None:
        audit_section = {}
    if not isinstance(audit_section, dict):
        raise ValueError(
            "audit config section must be a mapping, got "
            f"{type(audit_section).__name__}"
        )
    warn_unknown_keys(audit_section, logger=logger, section="audit")

    def audit_pick(key: str, default: Any) -> Any:
        env_key = f"{ENV_PREFIX}AUDIT_{key.upper()}"
        if env.get(env_key):
            return env[env_key]
        if audit_section.get(key) is not None:
            return audit_section[key]
        return default

    audit = AuditSettings(
        enabled=_parse_bool(
            audit_pick("enabled", False), f"{ENV_PREFIX}AUDIT_ENABLED"
        ),
        path=_parse_audit_path(audit_pick("path", "stdout")),
        include_text=_parse_bool(
            audit_pick("include_text", False),
            f"{ENV_PREFIX}AUDIT_INCLUDE_TEXT",
        ),
        access_log=_parse_bool(
            audit_pick("access_log", False), f"{ENV_PREFIX}AUDIT_ACCESS_LOG"
        ),
    )
    if audit.enabled and audit.path not in AUDIT_STREAM_TARGETS:
        parent = Path(audit.path).parent
        if parent.exists() and not parent.is_dir():
            raise ValueError(
                f"audit path parent is not a directory: {audit.path!r}"
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
        max_body_bytes=max_body_bytes,
        config_path=resolved_path,
        normalization=normalization,
        fast_path=fast_path,
        deep_path=deep_path,
        canary=canary,
        rate_limit=rate_limit,
        audit=audit,
    )
    logger.debug(
        "Loaded settings: upstream=%s host=%s port=%s timeout=%s "
        "normalization.mode=%s fast_path.enabled=%s "
        "fast_path.block=%s fast_path.warn=%s "
        "deep_path.enabled=%s deep_path.model_path=%s "
        "canary.enabled=%s canary.action=%s canary.secret=%s",
        settings.upstream_url,
        settings.host,
        settings.port,
        settings.request_timeout,
        settings.normalization.mode,
        settings.fast_path.enabled,
        settings.fast_path.block_threshold,
        settings.fast_path.warn_threshold,
        settings.deep_path.enabled,
        settings.deep_path.model_path,
        settings.canary.enabled,
        settings.canary.action,
        "set" if settings.canary.secret else "ephemeral",
    )
    return settings
