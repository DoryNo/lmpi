# LMPI — LLM Prompt-Injection Firewall

Transparent proxy that protects LLM applications from prompt injection and system prompt leakage. Drop-in replacement for OpenAI API `base_url` — your app keeps working, but every prompt goes through a three-layer detection pipeline.

> **v1 MVP Status:** In development. See [PLAN.md](PLAN.md) for timeline, [ROADMAP.md](ROADMAP.md) for future features.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LMPI Proxy                               │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌─────────┐ │
│  │  Ingress  │───▶│ Fast Path │───▶│Deep Path  │───▶│ Decision│ │
│  │Normalize  │    │ (Regex)   │    │  (ML)     │    │         │ │
│  └──────────┘    └───────────┘    └───────────┘    └─────────┘ │
│       │               │                │                │       │
│       ▼               ▼                ▼                ▼       │
│  NFKC/zero-      Jailbreak        DeBERTa-        block /      │
│  width/base64    patterns         v3-base          warn /       │
│  hex/rot13       (weighted)       (ONNX RT)        log-only    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Canary Token Detection                       │   │
│  │  HMAC token in system prompt → grep in output stream      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Target LLM API  │
                    │  (OpenAI, etc.)  │
                    └──────────────────┘
```

## Benchmark Results

> ⚠️ Results will be published after v1 completion. Metrics will include:

| Metric | Value | Notes |
|--------|-------|-------|
| Attack Detection Rate (TPR) | TBD | JailbreakBench + HarmBench subset |
| False Positive Rate | TBD | Clean prompts dataset |
| p50 Latency Overhead | TBD | Proxy vs direct |
| p95 Latency Overhead | TBD | Proxy vs direct |

**Methodology:** Frozen eval set, not mixed with tuning data. Model version and config hash documented for reproducibility.

## Quickstart

```bash
# Docker one-liner (coming soon)
docker run -p 8080:8080 -e UPSTREAM_URL=https://api.openai.com lmpi:latest

# Or point your OpenAI client to LMPI
export OPENAI_BASE_URL=http://localhost:8080/v1
```

## Quickstart (dev)

Requires Python 3.11+.

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the proxy (forwards to LMPI_UPSTREAM_URL, default https://api.openai.com)
uvicorn src.main:app --host 0.0.0.0 --port 8080
# ...or let the proxy read LMPI_HOST / LMPI_PORT / LMPI_UPSTREAM_URL itself:
python -m src.main

# Docker Compose alternative
docker compose up --build
```

Try it (streaming — `-N` disables curl buffering so you see SSE chunks live):

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "stream": true, "messages": [{"role": "user", "content": "Hello!"}]}'
```

Non-streaming: drop `"stream": true` (and `-N`). Health check: `curl http://localhost:8080/health`.

Configuration (env vars override `config.yaml`):

| Env var | Default | Purpose |
|---------|---------|---------|
| `LMPI_UPSTREAM_URL` | `https://api.openai.com` | Upstream LLM API base URL |
| `LMPI_HOST` / `LMPI_PORT` | `0.0.0.0` / `8080` | Proxy bind address |
| `LMPI_REQUEST_TIMEOUT` | `300.0` | Read timeout, seconds |
| `LMPI_CONFIG_PATH` | — | Path to a YAML config file |

Run tests (no real network — upstream is mocked):

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Limitations (v1)

Honest list of what v1 does NOT include:

- **No tool call firewall** — SSRF/path traversal detection for function calls is not implemented
- **No DLP filter** — PII/sensitive data detection not included
- **No fine-tuned model** — using pretrained `deberta-v3-base-prompt-injection`, not optimized for LMPI-specific patterns
- **No Redis/Prometheus/Grafana** — in-memory state, basic logging only
- **No Rust/Go port** — pure Python for v1
- **Single upstream** — no load balancing or failover
- **No auth** — proxy itself doesn't authenticate clients (use network-level auth)

See [ROADMAP.md](ROADMAP.md) for planned features.

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| v1.0 | Core detection pipeline + benchmark | 🔨 In progress |
| v1.1 | Rate limiting, audit log, hot-reload | 📋 Planned |
| v2.0 | Tool call firewall (SSRF, path traversal) | 📋 Planned |
| v2.5 | DLP filter (PII detection, masking) | 📋 Planned |
| v3.0 | Fine-tuned model | 📋 Planned |
| v3.5 | Redis + Prometheus + Grafana | 📋 Planned |
| v4.0 | Rust/Go port | 📋 Planned |

## Project Structure

```
lmpi/
├── src/
│   ├── main.py              # FastAPI app
│   ├── proxy.py             # httpx proxy + SSE streaming
│   ├── config.py            # Configuration
│   ├── normalization/       # Ingress normalization
│   ├── fast_path/           # Regex/heuristic detection
│   ├── deep_path/           # ML classifier (ONNX)
│   └── canary/              # Canary token detection
├── tests/                   # Unit + integration tests
├── benchmarks/              # Frozen eval set + runner
├── models/                  # Quantized ONNX models
├── IDEA.md                  # Core idea and positioning
├── PLAN.md                  # Development plan (v1)
├── ROADMAP.md               # Future features
└── AGENTS.md                # Task breakdown for agents
```

## License

MIT
