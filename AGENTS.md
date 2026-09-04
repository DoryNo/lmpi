# LMPI — Development Agents / Task Breakdown

Структура задач для делегирования агентам (Copilot sessions). Каждый агент — изолированная задача с чётким scope.

---

## Agent 1: Core Proxy Setup

**Задача:** Создать базовый FastAPI прокси с поддержкой SSE-стриминга

**Статус:** ✅ done — PR #1

**Scope:**
- FastAPI app skeleton
- Эндпоинт `/v1/chat/completions` (POST)
- Проксификация через httpx к configurable upstream
- SSE-стриминг (Server-Sent Events passthrough)
- Health-check `/health`
- Конфиг через env vars + YAML
- Dockerfile + docker-compose.yaml
- Basic integration test

**Выход:** Рабочий прокси, который прозрачно форвардит запросы к upstream LLM

---

## Agent 2: Ingress Normalization

**Задача:** Модуль нормализации входного текста

**Статус:** ✅ done — PR #2

**Scope:**
- `normalization/unicode.py` — NFKC, zero-width chars, control chars
- `normalization/encoding.py` — base64, hex, rot13 decode-and-recheck
- `normalization/delimiters.py` — pseudo-system delimiter neutralization
- Comprehensive unit tests для каждого модуля
- Benchmark: latency overhead measurement

**Выход:** Модуль с 90%+ coverage, все edge cases покрыты тестами

---

## Agent 3: Fast Path Detection

**Задача:** Regex/heuristic движок для известных jailbreak-паттернов

**Статус:** ✅ done — PR #3

**Scope:**
- `fast_path/patterns.py` — паттерны на:
  - "Ignore previous instructions" вариации
  - Role-play jailbreaks (DAN, Developer Mode)
  - System/Assistant prompt injection
  - Multi-language patterns
  - Encoding-based bypasses
- Weighted scoring система
- Unit tests с positive/negative cases
- Threshold tuning документация

**Выход:** Fast path модуль с documented detection rate на известных паттернах

---

## Agent 4: Canary Token System

**Задача:** Система обнаружения утечки system prompt через canary tokens

**Статус:** ✅ done — PR #5

**Scope:**
- HMAC-based token generation (per-session secret)
- Внедрение canary в system prompt при проксификации
- Детекция canary в исходящем стриме ответа
- Alert/logging при обнаружении
- Unit tests (generation, injection, detection)

**Выход:** Рабочий canary detection с тестами

---

## Agent 5: Deep Path (ML Classifier)

**Задача:** Интеграция quantized ML-классификатора через ONNX Runtime

**Статус:** ✅ done — PR #4 *(delivered full-precision ONNX: no quantized export exists upstream — see PLAN.md week 3)*

**Scope:**
- Загрузка `protectai/deberta-v3-base-prompt-injection` (ONNX)
- Tokenization pipeline
- Inference pipeline с threshold
- Graceful degradation если модель недоступна
- Latency benchmark
- Unit tests (mock inference)

**Выход:** ML классификатор, интегрированный в pipeline

---

## Agent 6: Detection Pipeline Orchestrator

**Задача:** Объединение всех модулей в единый pipeline

**Статус:** ✅ done — integrated with the stage PRs (#3–#5); no separate PR

**Scope:**
- Orchestrator: normalization → fast path → deep path → decision
- Configurable actions: block / warn / log-only
- Structured logging (JSON) для detection events
- Integration tests (end-to-end через прокси)
- Error handling + fallback behavior

**Выход:** Единый detection pipeline, протестированный end-to-end

---

## Agent 7: Benchmark Framework

**Задача:** Воспроизводимый бенчмарк с frozen eval set

**Статус:** ✅ done — PR #6

**Scope:**
- Eval dataset curation:
  - JailbreakBench subset (~200 attacks)
  - HarmBench subset (~100 attacks)
  - Clean dataset (~300 benign prompts)
- `benchmarks/run_benchmark.py` — автоматический прогон
- Метрики: TPR, FPR, p50/p95 latency
- Per-module breakdown
- Frozen eval set documentation
- Reproducibility (seed, model version, config hash)

**Выход:** Benchmark script + результаты в таблице для README

---

## Agent 8: README + Documentation

**Задача:** Финальный README с честными метриками

**Статус:** ✅ done — this PR (README polish, verified quickstart, demo examples, docs consistency, CI)

**Scope:**
- Что это и зачем (1 абзац)
- Архитектура (ASCII-диаграмма)
- Результаты бенчмарка (таблица)
- Quickstart / Docker one-liner
- Limitations (честный список)
- Roadmap (ссылка на ROADMAP.md)
- Contributing guide (если open source)

**Выход:** README, который честно описывает проект и его метрики

---

## Execution Order

```
Agent 1 (Core Proxy)
    ↓
Agent 2 (Ingress Normalization)  ← параллельно с Agent 3
Agent 3 (Fast Path)
    ↓
Agent 4 (Canary Tokens)  ← параллельно с Agent 5
Agent 5 (Deep Path)
    ↓
Agent 6 (Pipeline Orchestrator)  ← после агентов 2-5
    ↓
Agent 7 (Benchmark)  ← после Agent 6
    ↓
Agent 8 (README)  ← после Agent 7
```

---

## Dependency Graph

| Agent | Depends On | Blocks |
|-------|-----------|--------|
| 1 | — | 2, 3, 4, 5, 6 |
| 2 | 1 | 6 |
| 3 | 1 | 6 |
| 4 | 1 | 6 |
| 5 | 1 | 6 |
| 6 | 2, 3, 4, 5 | 7 |
| 7 | 6 | 8 |
| 8 | 7 | — |
