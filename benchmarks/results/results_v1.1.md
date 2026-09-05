# LMPI benchmark results

Frozen eval set — 200 attack prompts, 300 clean prompts. Manifest sha256: `146689f09e9bf394…`. Run finished 2026-09-04T20:40:56+00:00 with pipeline 0.1.0 (git 32ea00b9754d28b90b35c0780c628afde6f2d54d).

## Headline

| Metric | Attacks | Clean |
|--------|---------|-------|
| Items | 200 | 300 |
| Blocked (rate) | 78 (39.0%) | 9 (3.0%) |
| Warned, forwarded (rate) | 0 (0.0%) | 0 (0.0%) |

**TPR (attack detection rate)** = attacks blocked by the pipeline. **FPR (false positive rate)** = clean prompts blocked. The warn rate is reported separately because warned prompts are logged but still forwarded to the LLM.

## Detection rate by source

| Source | Split | Items | Blocked | Block rate | Warned |
|--------|-------|-------|---------|------------|--------|
| jbb_harmful | attack | 100 | 0 | 0.0% | 0 |
| wild_jailbreaks | attack | 100 | 78 | 78.0% | 0 |
| jbb_benign | clean | 100 | 1 | 1.0% | 0 |
| ultrachat | clean | 170 | 0 | 0.0% | 0 |
| tricky_benign | clean | 30 | 8 | 26.7% | 0 |

## Tuning / held-out partition

Deterministic 60/40 tuning/held-out split of the frozen set (seed 20260911, `benchmarks\eval_set\split.json`). All tuning decisions used the tuning partition only; held-out numbers below were produced by the single final run.

| Partition | Split | Items | Blocked | Block rate | Warned |
|-----------|-------|-------|---------|------------|--------|
| tuning | attack | 120 | 49 | 40.8% | 0 |
| tuning | clean | 180 | 8 | 4.4% | 0 |
| held_out | attack | 80 | 29 | 36.2% | 0 |
| held_out | clean | 120 | 1 | 0.8% | 0 |

## Per-stage attribution — attacks

| Stage | Count | Rate |
|-------|-------|------|
| Fast path block | 14 | 7.0% |
| Fast path warn (forwarded) | 0 | 0.0% |
| Deep path block | 75 | 37.5% |
| Deep path warn (forwarded) | 0 | 0.0% |
| Blocked by both fast and deep (overlap) | 11 | 5.5% |
| Fast path only | 3 | 1.5% |
| Deep path only | 64 | 32.0% |
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
| Pipeline end-to-end (all items) | 29.7 | 604.7 | 646.9 | 129.1 | 706.1 |
| Attack items | 27.5 | 610.7 | 655.6 | 163.1 | 706.1 |
| Clean items | 30.5 | 596.0 | 620.6 | 106.4 | 678.1 |

| Stage | p50 | p95 | mean |
|-------|-----|-----|------|
| normalization | 0.18 | 3.03 | 0.66 |
| fast_path | 0.21 | 5.74 | 1.20 |
| deep_path | 30.69 | 598.22 | 139.98 |

**What is measured:** `DetectionPipeline.process_request()` wall time — stage 1 normalization rewrite + stage 2 regex scoring + stage 3 ONNX inference (CPU, first-512-token truncation). No upstream LLM call, no network I/O: this is the per-request overhead LMPI adds in front of the target LLM.

## Reproducibility

- **pipeline version:** 0.1.0
- **git commit:** 32ea00b9754d28b90b35c0780c628afde6f2d54d
- **python:** 3.13.14
- **onnxruntime:** 1.29.0
- **tokenizers:** 0.22.2
- **datasets:** 5.0.1
- **model:** deberta-v3-base-prompt-injection-v2 (full-precision)
- **model sha256:** `f0ea7f239f765aedbde7c9e1…`
- **tokenizer sha256:** `752fe5f0d5678ad563e1bd2e…`
- **manifest sha256:** `146689f09e9bf3940c1cce58…`
- **selection seed:** 20260905

Per-item records (IDs + decisions + timings, no prompt texts) are in the companion `results_v1.1.json`.
