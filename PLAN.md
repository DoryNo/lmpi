# LMPI — Development Plan (v1 MVP)

## Timeline: 3–4 недели

| Неделя | Фокус | Deliverable |
|--------|-------|-------------|
| 1 | Core proxy + Ingress Normalization | FastAPI прокси с /v1/chat/completions, SSE streaming, NFKC/zero-width/base64 decode |
| 2 | Fast Path + Canary Tokens | Regex/heuristic patterns, HMAC canary detection, unit tests |
| 3 | Deep Path (ML) | ONNX Runtime интеграция, quantized DeBERTa classifier, inference pipeline |
| 4 | Benchmark + README | Frozen eval set, прогон метрик, финальный README с таблицами |

---

## Неделя 1: Core Proxy + Ingress Normalization

### 1.1 FastAPI Proxy Skeleton
- [ ] FastAPI app с эндпоинтом `/v1/chat/completions`
- [ ] Проксификация через `httpx` к upstream LLM (configurable base_url)
- [ ] Поддержка SSE-стриминга (Server-Sent Events)
- [ ] Конфиг через env vars / YAML-файл
- [ ] Health-check эндпоинт `/health`
- [ ] Docker + docker-compose

### 1.2 Ingress Normalization Module
- [ ] NFKC normalization (Unicode)
- [ ] Удаление zero-width characters (U+200B, U+200C, U+200D, U+FEFF, etc.)
- [ ] Удаление control characters (кроме \n, \t)
- [ ] Base64 decode-and-recheck (обнаружение base64-encoded injection)
- [ ] Hex decode-and-recheck
- [ ] ROT13 decode-and-recheck
- [ ] Нейтрализация псевдо-системных разделителей (fake system/user markers)
- [ ] Unit tests для каждого модуля нормализации

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

### 2.1 Fast Path (Regex/Heuristics)
- [ ] Паттерны на известные jailbreak-конструкции:
  - "Ignore previous instructions"
  - "You are now DAN/Developer Mode"
  - "Pretend you are..."
  - "System: ..." / "Assistant: ..." инъекции
  - Multi-language jailbreak patterns
  - Encoding-based bypass attempts
- [ ] Scoring система (weighted patterns → composite score)
- [ ] Threshold tuning с метриками TP/FP
- [ ] Unit tests с positive/negative cases

### 2.2 Canary Token Detection
- [ ] HMAC-based canary token generation (per-session secret)
- [ ] Внедрение canary в system prompt при проксификации
- [ ] Детекция canary в исходящем стриме ответа модели
- [ ] Alert/logging при обнаружении утечки
- [ ] Unit tests

### 2.3 Detection Pipeline
- [ ] Orchestrator: ingress normalization → fast path → deep path → decision
- [ ] Конфигурируемые action при обнаружении: block / warn / log-only
- [ ] Structured logging (JSON) для каждого detection event

---

## Неделя 3: Deep Path (ML Classifier)

### 3.1 Model Integration
- [ ] Выбор модели: `protectai/deberta-v3-base-prompt-injection` (ONNX)
- [ ] ONNX Runtime интеграция с quantized inference
- [ ] Tokenization pipeline (HuggingFace tokenizers)
- [ ] Batch inference для эффективности
- [ ] Graceful degradation если модель недоступна

### 3.2 Classifier Pipeline
- [ ] Pre-processing: ingress normalization → текст для классификатора
- [ ] Inference: вероятность injection (0.0–1.0)
- [ ] Post-processing: threshold-based decision
- [ ] Latency optimization (caching, batching)

### 3.3 Honest Documentation
- [ ] В README честно указать: модель предобученная, не fine-tuned на наших данных
- [ ] Указать limitations модели (known false positive patterns, language coverage)
- [ ] Ссылка на оригинальную модель и её benchmarks

---

## Неделя 4: Benchmark + README

### 4.1 Evaluation Dataset
- [ ] JailbreakBench — атаки (subset, ~200 примеров)
- [ ] HarmBench — дополнительные атаки (subset, ~100 примеров)
- [ ] Clean dataset — benign prompts (~300 примеров) для FP measurement
- [ ] Frozen eval set — зафиксирован, не меняется между раундами
- [ ] Явная документация: source, size, split methodology

### 4.2 Benchmark Pipeline
- [ ] `run_benchmark.py` — автоматический прогон eval set через pipeline
- [ ] Метрики:
  - Attack Detection Rate (True Positive Rate)
  - False Positive Rate на clean dataset
  - p50 / p95 latency overhead (proxy vs direct)
- [ ] Per-module breakdown (normalization catches X, fast path catches Y, deep path catches Z)
- [ ] Reproducibility: seed, model version, config hash

### 4.3 README Finalization
- [ ] Что это и зачем (1 абзац)
- [ ] Архитектура (ASCII-диаграмма: ingress → fast path → deep path → target LLM)
- [ ] Результаты бенчмарка (таблица TP/FP/latency)
- [ ] Quickstart / Docker one-liner
- [ ] Limitations (честный список)
- [ ] Roadmap (ссылка на ROADMAP.md)

### 4.4 Quality Gates
- [ ] Все unit tests проходят
- [ ] Benchmark воспроизводится (запуск дважды → одинаковые цифры)
- [ ] Docker build работает
- [ ] README соответствует реальности (нет unreferenced claims)
