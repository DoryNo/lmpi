# LMPI benchmark results

Frozen eval set — 200 attack prompts, 300 clean prompts. Manifest sha256: `146689f09e9bf394…`. Run finished 2026-09-04T20:06:43+00:00 with pipeline 0.1.0 (git cd6b7e324b0e2ab266d86226e50b62d724465df0).

## Headline

| Metric | Attacks | Clean |
|--------|---------|-------|
| Items | 200 | 300 |
| Blocked (rate) | 73 (36.5%) | 9 (3.0%) |
| Warned, forwarded (rate) | 3 (1.5%) | 0 (0.0%) |

**TPR (attack detection rate)** = attacks blocked by the pipeline. **FPR (false positive rate)** = clean prompts blocked. The warn rate is reported separately because warned prompts are logged but still forwarded to the LLM.

## Detection rate by source

| Source | Split | Items | Blocked | Block rate | Warned |
|--------|-------|-------|---------|------------|--------|
| jbb_harmful | attack | 100 | 0 | 0.0% | 0 |
| wild_jailbreaks | attack | 100 | 73 | 73.0% | 3 |
| jbb_benign | clean | 100 | 1 | 1.0% | 0 |
| ultrachat | clean | 170 | 0 | 0.0% | 0 |
| tricky_benign | clean | 30 | 8 | 26.7% | 0 |

## Per-stage attribution — attacks

| Stage | Count | Rate |
|-------|-------|------|
| Fast path block | 8 | 4.0% |
| Fast path warn (forwarded) | 0 | 0.0% |
| Deep path block | 72 | 36.0% |
| Deep path warn (forwarded) | 3 | 1.5% |
| Blocked by both fast and deep (overlap) | 7 | 3.5% |
| Fast path only | 1 | 0.5% |
| Deep path only | 65 | 32.5% |
| Normalization findings (rewrite mode, non-blocking) | 14 | 7.0% |

## Per-stage attribution — clean prompts

| Stage | Count | Rate |
|-------|-------|------|
| Fast path block | 1 | 0.3% |
| Fast path warn (forwarded) | 0 | 0.0% |
| Deep path block | 9 | 3.0% |
| Deep path warn (forwarded) | 0 | 0.0% |
| Blocked by both fast and deep (overlap) | 1 | 0.3% |
| Fast path only | 0 | 0.0% |
| Deep path only | 8 | 2.7% |
| Normalization findings (rewrite mode, non-blocking) | 5 | 1.7% |

## Latency (per request, CPU, no LLM call)

| Measured over | p50 | p95 | p99 | mean | max |
|---------------|-----|-----|-----|------|-----|
| Pipeline end-to-end (all items) | 29.5 | 557.5 | 574.2 | 123.8 | 588.2 |
| Attack items | 28.6 | 562.6 | 578.4 | 162.9 | 588.2 |
| Clean items | 29.7 | 547.9 | 569.7 | 97.7 | 575.0 |

| Stage | p50 | p95 | mean |
|-------|-----|-----|------|
| normalization | 0.17 | 2.90 | 0.64 |
| fast_path | 0.18 | 4.39 | 0.99 |
| deep_path | 29.94 | 550.66 | 129.07 |

**What is measured:** `DetectionPipeline.process_request()` wall time — stage 1 normalization rewrite + stage 2 regex scoring + stage 3 ONNX inference (CPU, first-512-token truncation). No upstream LLM call, no network I/O: this is the per-request overhead LMPI adds in front of the target LLM.

## Reproducibility

- **pipeline version:** 0.1.0
- **git commit:** cd6b7e324b0e2ab266d86226e50b62d724465df0
- **python:** 3.13.14
- **onnxruntime:** 1.29.0
- **tokenizers:** 0.22.2
- **datasets:** 5.0.1
- **model:** deberta-v3-base-prompt-injection-v2 (full-precision)
- **model sha256:** `f0ea7f239f765aedbde7c9e1…`
- **tokenizer sha256:** `752fe5f0d5678ad563e1bd2e…`
- **manifest sha256:** `146689f09e9bf3940c1cce58…`
- **selection seed:** 20260905

Per-item records (IDs + decisions + timings, no prompt texts) are in the companion `results_v1.0.json`.
