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

## Ingress Normalization

The first pipeline stage (`src/normalization/`) rewrites every user message before detection runs:

- **Unicode cleanup** — NFKC, zero-width/invisible chars (U+200B, U+FEFF, soft hyphens…), bidi controls and non-whitespace control chars removed, so hidden characters can no longer split keywords or corrupt parsing.
- **Decode-and-recheck** — base64 / hex blobs are decoded and inlined so later stages see the payload; ROT13 runs are rewritten only when the decode contains a suspicious marker (`ignore`, `jailbreak`, …). Malformed blobs are skipped, never crash.
- **Delimiter neutralization** — pseudo-system tokens (`<|im_start|>`, `### System`, `[INST]`, line-start `System:` …) are replaced with visibly-escaped markers (`⟦fake-system⟧`), surgically: prose and code like `os.system("ls")` or `### System requirements` are left untouched.
- Every rewrite is recorded as a structured `Finding`; by default findings are logged and the payload is rewritten (`LMPI_NORMALIZATION_MODE=rewrite`; `block` returns 403, `log` is observe-only).

## Fast Path

The second pipeline stage (`src/fast_path/`) is a regex/heuristic detector that runs on the **normalized** text produced by stage 1:

- **Five categories with weighted patterns** — instruction override (~0.9), system prompt extraction (~0.9), roleplay jailbreaks (~0.8), fake role injection (~0.7), obfuscation markers (~0.5); a handful of Russian variants of the top patterns included (deliberately limited scope).
- **Noisy-OR composite scoring** — `score = 1 − Π(1 − wᵢ)`: multiple weak signals stack (three 0.5-weight obfuscation hits → 0.875), so evasion used together still blocks; duplicate hits of the same pattern count once.
- **Thresholds** — `score ≥ 0.75` → block (HTTP 403); `0.4 ≤ score < 0.75` → warn (logged, forwarded); configurable via `LMPI_FAST_PATH_BLOCK_THRESHOLD` / `LMPI_FAST_PATH_WARN_THRESHOLD`.
- **False-positive controls** — word-boundary anchoring, structural gating (a persona switch alone or a restriction-lift phrase alone never scores — e.g. "Write a story where a robot must answer without restrictions" stays clean), benign collocations excluded ("ignore previous test results"), and quoted mentions (`"ignore all previous instructions"` discussed in a security course) demoted to zero weight.
- **Honest caveat:** all weights are heuristic priors, not measurements — they will be tuned against the frozen benchmark eval set (see Benchmark Results below).

## Deep Path (ML)

The third pipeline stage (`src/deep_path/`) is a neural prompt-injection classifier that runs **only when the fast path did not already block** and on the **normalized** text (stage 1 output):

- **Model** — [`protectai/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2), ONNX export, inference via `onnxruntime` + the lightweight `tokenizers` library (no `transformers`/`torch`). The softmax over the 2 logits gives an injection probability in `[0, 1]`; `score ≥ 0.75` → block (HTTP 403), `score ≥ 0.5` → warn (logged, forwarded).
- **Not fine-tuned by us (honesty note):** this is the *pretrained* classifier as published upstream — LMPI adds zero training data. Its known false-positive patterns and language coverage limits carry over (see its [model card](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) for the author's benchmarks).
- **Quantization status:** the upstream repo currently publishes **no** quantized ONNX variant, so the full-precision `model.onnx` (~740 MB) is used; if a `model_quantized.onnx` is present in the model dir, the backend picks it up automatically and logs `quantized: true`.
- **Enable it** (disabled by default):

  ```bash
  pip install -r requirements.txt
  python scripts/download_model.py     # downloads model.onnx + tokenizer into models/ (gitignored)
  export LMPI_DEEP_PATH_ENABLED=true
  ```

- **Graceful degradation** — model missing, or `onnxruntime` not installed → the stage is skipped with a one-time warning and the proxy keeps working (stages 1–2 remain active).
- **Input hygiene & truncation** — text is capped at `LMPI_DEEP_PATH_MAX_CHARS` (default 6000) before classification (capped requests are flagged in the log event), then token-truncated to the model's 512-token training window (standard **first-512** truncation: an injection payload near the end of a >512-token prompt can be missed — known trade-off).
- **Latency** — every decision event logs `latency_ms` (tokenize + ONNX inference); on CPU, single-text inference is roughly 20–30 ms, feeding the benchmark's per-stage breakdown.

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

# 2. (optional) Download the deep-path ML model, then enable it
python scripts/download_model.py
export LMPI_DEEP_PATH_ENABLED=true

# 3. Run the proxy (forwards to LMPI_UPSTREAM_URL, default https://api.openai.com)
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
| `LMPI_NORMALIZATION_MODE` | `rewrite` | Normalization action: `rewrite` / `block` / `log` |
| `LMPI_NORMALIZATION_UNICODE` | `true` | NFKC + zero-width/bidi/control cleanup |
| `LMPI_NORMALIZATION_BASE64` | `true` | base64 decode-and-recheck |
| `LMPI_NORMALIZATION_HEX` | `true` | hex decode-and-recheck |
| `LMPI_NORMALIZATION_ROT13` | `true` | ROT13 decode (marker-gated) |
| `LMPI_NORMALIZATION_DELIMITERS` | `true` | Pseudo-system delimiter neutralization |
| `LMPI_FAST_PATH_ENABLED` | `true` | Fast-path regex/heuristic stage on/off |
| `LMPI_FAST_PATH_BLOCK_THRESHOLD` | `0.75` | Noisy-OR score at/above which requests are blocked |
| `LMPI_FAST_PATH_WARN_THRESHOLD` | `0.4` | Score at/above which requests are logged (warn) but forwarded |
| `LMPI_DEEP_PATH_ENABLED` | `false` | Deep-path ML stage on/off (download model first) |
| `LMPI_DEEP_PATH_MODEL_PATH` | `models/deberta-v3-base-prompt-injection-v2` | Directory with `model.onnx` + `tokenizer.json` |
| `LMPI_DEEP_PATH_BLOCK_THRESHOLD` | `0.75` | Injection probability at/above which requests are blocked |
| `LMPI_DEEP_PATH_WARN_THRESHOLD` | `0.5` | Probability at/above which requests are logged (warn) but forwarded |
| `LMPI_DEEP_PATH_MAX_CHARS` | `6000` | Cap on text length handed to the classifier |

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
- **Deep path truncation** — the classifier sees only the first 512 tokens (its training window); injection payloads near the end of very long prompts can be missed
- **No quantized deep-path model** — upstream ships no quantized ONNX export, so the full-precision ~740 MB model is downloaded
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
├── scripts/
│   └── download_model.py    # Downloads the deep-path ONNX model (gitignored)
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
