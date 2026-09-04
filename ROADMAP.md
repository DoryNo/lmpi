# LMPI — Roadmap

## v1.0 — MVP (✅ доставлено)

**Фокус:** Prompt Injection Detection + Canary Token Detection

Что входит:
- Ingress normalization (NFKC, zero-width, base64/hex/rot13, pseudo-system delimiters)
- Fast path (regex/heuristic patterns)
- Deep path (DeBERTa classifier via ONNX Runtime — full-precision; upstream publishes no quantized export)
- Canary token detection (HMAC-based system prompt leakage detection)
- Transparent proxy (/v1/chat/completions + SSE streaming)
- Frozen benchmark с честными метриками (TP/FP/latency)

Threshold tuning: сознательно **не выполнено** в v1 — поставляемые пороги
это baseline-as-shipped значения, зафиксированные до прогона бенчмарка,
чтобы метрики оставались out-of-sample. Тюнинг по eval set — первый пункт
v1.1.

Что НЕ входит (осознанно):
- Tool call firewall
- DLP-фильтр
- Своя fine-tuned модель
- Redis / Prometheus / Grafana
- Rust/Go порт

---

## v1.1 — Hardening

- [ ] Threshold tuning against the frozen eval set — *deliberately skipped in v1 (conscious decision: keep the shipped benchmark numbers baseline-as-shipped / out-of-sample); first v1.1 item, documented as its own iteration*
- [ ] Rate limiting (per-client, per-API-key)
- [ ] Request/response logging с redaction
- [ ] Structured audit log (JSON lines)
- [ ] Configuration validation + hot-reload
- [ ] Graceful shutdown + healthcheck improvements
- [ ] Additional encoding detection (punycode, HTML entities, URL encoding)
- [ ] Multi-language jailbreak pattern expansion

---

## v2.0 — Tool Call Firewall

**Фокус:** Защита от атак через tool/function calling

- [ ] Tool call validation (SSRF detection)
- [ ] Path traversal detection в file操作ах
- [ ] Parameter sanitization для function calls
- [ ] Allowlist/denylist для tool names
- [ ] Tool call rate limiting
- [ ] Benchmark: tool-call-specific attack dataset

---

## v2.5 — DLP Filter

**Фокус:** Data Loss Prevention — предотвращение утечки чувствительных данных

- [ ] PII detection (email, phone, SSN, credit card patterns)
- [ ] Custom regex patterns для организации-specific данных
- [ ] Entity masking (замена PII на placeholders)
- [ ] Per-field allow/deny rules
- [ ] DLP benchmark с synthetic PII dataset

---

## v3.0 — Fine-Tuned Model

**Фокус:** Собственная модель, обученная на специфичных для LMPI данных

- [ ] Dataset collection: атаки, которые не ловит текущая модель
- [ ] Fine-tuning pipeline (LoRA на DeBERTa или аналог)
- [ ] A/B testing framework: base model vs fine-tuned
- [ ] Continuous learning loop: new attacks → dataset → retrain
- [ ] Model versioning + rollback

---

## v3.5 — Observability Stack

- [ ] Redis для distributed rate limiting + кэширования
- [ ] Prometheus метрики (request count, detection rate, latency percentiles)
- [ ] Grafana dashboards
- [ ] Alerting rules (high FP rate, latency spike, detection spike)
- [ ] OpenTelemetry tracing

---

## v4.0 — Rust/Go Port

**Фокус:** Performance-critical путь на системном языке

- [ ] Core detection pipeline на Rust (или Go)
- [ ] Python fallback для ML inference
- [ ] FFI bindings для ONNX Runtime
- [ ] Benchmark: latency comparison Python vs Rust
- [ ] Binary distribution (single executable)

---

## Принципы приоритизации

1. **Измеримость** — каждая фича должна иметь бенчмарк-метрику
2. **Обратная совместимость** — новые версии не ломают существующий API
3. **Честность** — если что-то не работает идеально, пишем об этом
4. **MVP-first** — маленький завершённый шаг лучше большого недоделанного
