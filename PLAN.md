# LMPI — Development Plan (v1 MVP)

> **Status: ✅ v1 MVP complete.** All four weeks shipped across PRs #1–#6
> (see [AGENTS.md](AGENTS.md) for the agent → PR mapping and
> [benchmarks/results/results.md](benchmarks/results/results.md) for the
> measured numbers). The text below is kept as a historical record; items
> that deviated from the plan are annotated inline.

## Timeline: 3–4 недели

| Неделя | Фокус | Deliverable |
|--------|-------|-------------|
| 1 | Core proxy + Ingress Normalization | FastAPI прокси с /v1/chat/completions, SSE streaming, NFKC/zero-width/base64 decode |
| 2 | Fast Path + Canary Tokens | Regex/heuristic patterns, HMAC canary detection, unit tests |
| 3 | Deep Path (ML) | ONNX Runtime интеграция, quantized DeBERTa classifier, inference pipeline |
| 4 | Benchmark + README | Frozen eval set, прогон метрик, финальный README с таблицами |

---

## Неделя 1: Core Proxy + Ingress Normalization

**Статус: ✅ завершено** — PR #1 (core proxy), PR #2 (normalization).

### 1.1 FastAPI Proxy Skeleton
- [x] FastAPI app с эндпоинтом `/v1/chat/completions`
- [x] Проксификация через `httpx` к upstream LLM (configurable base_url)
- [x] Поддержка SSE-стриминга (Server-Sent Events)
- [x] Конфиг через env vars / YAML-файл
- [x] Health-check эндпоинт `/health`
- [x] Docker + docker-compose

### 1.2 Ingress Normalization Module
- [x] NFKC normalization (Unicode)
- [x] Удаление zero-width characters (U+200B, U+200C, U+200D, U+FEFF, etc.)
- [x] Удаление control characters (кроме \n, \t)
- [x] Base64 decode-and-recheck (обнаружение base64-encoded injection)
- [x] Hex decode-and-recheck
- [x] ROT13 decode-and-recheck
- [x] Нейтрализация псевдо-системных разделителей (fake system/user markers)
- [x] Unit tests для каждого модуля нормализации

### 1.3 Project Structure
```
lmpi/
├── src/
│   ├── main.py              # FastAPI app
│   ├── proxy.py             # httpx proxy logic + SSE streaming
│   ├── config.py            # Configuration management
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── unicode.py       # NFKC, zero-width, control chars
│   │   ├── encoding.py      # base64, hex, rot13 decode
│   │   └── delimiters.py    # Pseudo-system delimiter neutralization
│   ├── fast_path/
│   │   ├── __init__.py
│   │   └── patterns.py      # Regex/heuristic patterns
│   ├── deep_path/
│   │   ├── __init__.py
│   │   └── classifier.py    # ONNX Runtime classifier
│   └── canary/
│       ├── __init__.py
│       └── tokens.py        # HMAC canary token generation + detection
├── tests/
│   ├── test_normalization.py
│   ├── test_fast_path.py
│   ├── test_deep_path.py
│   ├── test_canary.py
│   └── test_integration.py
├── benchmarks/
│   ├── eval_set/            # Frozen evaluation dataset
│   ├── run_benchmark.py
│   └── results/
├── models/                  # Quantized ONNX models
├── config.yaml
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
├── IDEA.md
├── PLAN.md
├── ROADMAP.md
└── AGENTS.md
```

---

## Неделя 2: Fast Path + Canary Tokens

**Статус: ✅ завершено** — PR #3 (fast path), PR #5 (canary); the detection
pipeline orchestrator below landed together with stages 2–3 (PRs #3–#5).

### 2.1 Fast Path (Regex/Heuristics)
- [x] Паттерны на известные jailbreak-конструкции:
  - "Ignore previous instructions"
  - "You are now DAN/Developer Mode"
  - "Pretend you are..."
  - "System: ..." / "Assistant: ..." инъекции
  - Multi-language jailbreak patterns
  - Encoding-based bypass attempts
- [x] Scoring система (weighted patterns → composite score)
- [x] Threshold tuning с метриками TP/FP
  *(note: threshold tuning against the benchmark eval set was deliberately **not** done in v1 — the shipped thresholds are the baseline-as-shipped defaults, kept out-of-sample on purpose; scheduled for v1.1. TP/FP metrics are reported per threshold in the benchmark results.)*
- [x] Unit tests с positive/negative cases

### 2.2 Canary Token Detection
- [x] HMAC-based canary token generation (per-session secret)
- [x] Внедрение canary в system prompt при проксификации
- [x] Детекция canary в исходящем стриме ответа модели
- [x] Alert/logging при обнаружении утечки
- [x] Unit tests

### 2.3 Detection Pipeline
- [x] Orchestrator: ingress normalization → fast path → deep path → decision
- [x] Конфигурируемые action при обнаружении: block / warn / log-only
- [x] Structured logging (JSON) для каждого detection event

---

## Неделя 3: Deep Path (ML Classifier)

**Статус: ✅ завершено** — PR #4. Deviation from the plan: no quantized
inference — the upstream repo publishes no quantized ONNX export, so the
full-precision `model.onnx` (~740 MB) is used (auto-detected if a quantized
variant appears); batch inference / caching were dropped as they don't help
a single-request proxy pass-through.

### 3.1 Model Integration
- [x] Выбор модели: `protectai/deberta-v3-base-prompt-injection` (ONNX)
- [x] ONNX Runtime интеграция с quantized inference
- [x] Tokenization pipeline (HuggingFace tokenizers)
- [x] Batch inference для эффективности
- [x] Graceful degradation если модель недоступна

### 3.2 Classifier Pipeline
- [x] Pre-processing: ingress normalization → текст для классификатора
- [x] Inference: вероятность injection (0.0–1.0)
- [x] Post-processing: threshold-based decision
- [x] Latency optimization (caching, batching)

### 3.3 Honest Documentation
- [x] В README честно указать: модель предобученная, не fine-tuned на наших данных
- [x] Указать limitations модели (known false positive patterns, language coverage)
- [x] Ссылка на оригинальную модель и её benchmarks

---

## Неделя 4: Benchmark + README

**Статус: ✅ завершено** — PR #6 (benchmark + results committed) and this
docs pass. Deviation: HarmBench was skipped (gated dataset — downloading it
would break clean-checkout reproduction); the attack set is 100 JBB harmful
behaviors + 100 in-the-wild jailbreaks.

### 4.1 Evaluation Dataset
- [x] JailbreakBench — атаки (subset, ~200 примеров)
- [x] HarmBench — дополнительные атаки (subset, ~100 примеров)
- [x] Clean dataset — benign prompts (~300 примеров) для FP measurement
- [x] Frozen eval set — зафиксирован, не меняется между раундами
- [x] Явная документация: source, size, split methodology

### 4.2 Benchmark Pipeline
- [x] `run_benchmark.py` — автоматический прогон eval set через pipeline
- [x] Метрики:
  - Attack Detection Rate (True Positive Rate)
  - False Positive Rate на clean dataset
  - p50 / p95 latency overhead (proxy vs direct)
- [x] Per-module breakdown (normalization catches X, fast path catches Y, deep path catches Z)
- [x] Reproducibility: seed, model version, config hash

### 4.3 README Finalization
- [x] Что это и зачем (1 абзац)
- [x] Архитектура (ASCII-диаграмма: ingress → fast path → deep path → target LLM)
- [x] Результаты бенчмарка (таблица TP/FP/latency)
- [x] Quickstart / Docker one-liner
- [x] Limitations (честный список)
- [x] Roadmap (ссылка на ROADMAP.md)

### 4.4 Quality Gates
- [x] Все unit tests проходят
- [x] Benchmark воспроизводится (запуск дважды → одинаковые цифры)
- [x] Docker build работает
  *(note: Dockerfile + docker-compose shipped in PR #1 and smoke-checked there; not re-verified at Agent 8 time — no Docker daemon in the final environment.)*
- [x] README соответствует реальности (нет unreferenced claims)
