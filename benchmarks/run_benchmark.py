"""CLI entry point for the LMPI frozen-eval-set benchmark.

Runs every eval prompt from the frozen manifest through the full detection
pipeline (normalization rewrite + fast path + deep path with the real ONNX
backend) and writes:

- ``results.json`` — headline metrics, per-stage attribution, latency
  percentiles, per-item records (IDs + decisions + timings, NO prompt
  texts) and a reproducibility block (versions, model/tokenizer SHA-256,
  config snapshot, manifest SHA-256, seed, timestamps).
- ``results.md`` — the human-readable report committed to the repo.

Network use is limited to resolving the manifest's dataset rows (cached in
the gitignored ``benchmarks/.cache/``); everything else is local. No prompt
text is ever printed, logged, or written to results.

Selfcheck mode (--selfcheck) rebuilds the pipeline from scratch and runs a
small subset twice, asserting identical decisions (not latencies).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks import SEED  # noqa: E402
from benchmarks.hf_sources import configure_hf_caches, resolve_items  # noqa: E402
from benchmarks.manifest import load_manifest, sha256_file  # noqa: E402
from benchmarks.runner import (  # noqa: E402
    build_benchmark_pipeline,
    by_source_metrics,
    compare_decisions,
    compute_metrics,
    render_markdown,
    run_items,
)

RESULTS_SCHEMA = "lmpi-benchmark-results/1"

DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "eval_set" / "manifest.json"
DEFAULT_TRICKY = REPO_ROOT / "benchmarks" / "eval_set" / "tricky_benign.jsonl"
DEFAULT_MODEL = REPO_ROOT / "models" / "deberta-v3-base-prompt-injection-v2"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / ".cache"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "results" / "results.json"
DEFAULT_MD = REPO_ROOT / "benchmarks" / "results" / "results.md"

LATENCY_NOTE = (
    "Latency is perf_counter wall time around DetectionPipeline.process_request: "
    "stage 1 normalization rewrite + stage 2 regex fast-path scoring + stage 3 "
    "ONNX Runtime CPU inference (512-token truncation). No LLM call and no "
    "network I/O is included: this is the per-request overhead LMPI adds in "
    "front of the target LLM. Stage latencies are the instrumented per-stage "
    "times; the pipeline total may be slightly larger than the sum of stages."
)

CANARY_NOTE = (
    "Canary tokens are excluded: they detect exfiltration of the system prompt "
    "in the model's *output* stream, which is a different concern from "
    "classifying the *input* prompt. Attack-detection metrics here cover the "
    "input-side stages only (normalization, fast path, deep path)."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attacks", type=int, default=200, help="attack items to run")
    parser.add_argument("--clean", type=int, default=300, help="clean items to run")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tricky-path", type=Path, default=DEFAULT_TRICKY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--selfcheck-size", type=int, default=25)
    return parser.parse_args(argv)


def _git_fingerprint() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    candidates = [model_path / "model_quantized.onnx", model_path / "model.onnx"]
    weights = next((p for p in candidates if p.is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"no ONNX weights found under {model_path}")
    tokenizer = next(
        (
            model_path / name
            for name in ("tokenizer.json", "spm.model", "vocab.txt")
            if (model_path / name).is_file()
        ),
        None,
    )
    return {
        "name": model_path.name,
        "weights_file": weights.name,
        "quantized": weights.name != "model.onnx",
        "sha256": sha256_file(weights),
        "tokenizer_sha256": sha256_file(tokenizer) if tokenizer else None,
        "tokenizer_file": tokenizer.name if tokenizer else None,
    }


def _versions() -> dict[str, str]:
    import numpy
    import onnxruntime
    import tokenizers

    import src

    try:
        from datasets import __version__ as datasets_version
    except Exception:  # pragma: no cover
        datasets_version = "n/a"
    return {
        "lmpi_version": getattr(src, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "onnxruntime": onnxruntime.__version__,
        "tokenizers": tokenizers.__version__,
        "datasets": datasets_version,
        "numpy": numpy.__version__,
        "providers": list(onnxruntime.get_available_providers()),
    }


def run_selfcheck(args: argparse.Namespace) -> int:
    """Run a small subset twice with fresh pipelines; decisions must match."""
    manifest = load_manifest(args.manifest)
    size = max(1, args.selfcheck_size)
    attacks = list(manifest.attacks)[:size]
    clean = list(manifest.clean)[:size]

    configure_hf_caches(args.cache_dir)
    texts = resolve_items(
        manifest, [*attacks, *clean], args.tricky_path, cache_dir=args.cache_dir
    )
    items = [
        (item.id, "attack", texts[item.id]) for item in attacks
    ] + [(item.id, "clean", texts[item.id]) for item in clean]

    print(f"selfcheck: rebuilding pipeline and re-running {len(items)} items…")
    first = run_items(
        build_benchmark_pipeline(model_path=str(args.model_path)),
        items,
        warmup=1,
    )
    second = run_items(
        build_benchmark_pipeline(model_path=str(args.model_path)),
        items,
        warmup=1,
    )
    mismatches = compare_decisions(first, second)
    if mismatches:
        for message in mismatches[:10]:
            print(f"MISMATCH: {message}")
        print(f"selfcheck FAILED: {len(mismatches)} decision mismatches")
        return 1
    print(f"selfcheck OK: identical decisions on {len(items)} items (both runs)")
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    attacks = list(manifest.attacks)[: max(args.attacks, 0)]
    clean = list(manifest.clean)[: max(args.clean, 0)]
    if not attacks and not clean:
        print("nothing to run: --attacks and --clean are both 0", file=sys.stderr)
        return 1

    configure_hf_caches(args.cache_dir)
    resolved = resolve_items(
        manifest, [*attacks, *clean], args.tricky_path, cache_dir=args.cache_dir
    )
    items = [
        (item.id, "attack", resolved[item.id]) for item in attacks
    ] + [(item.id, "clean", resolved[item.id]) for item in clean]
    print(f"resolved {len(items)} items from the frozen manifest")

    try:
        bundle = build_benchmark_pipeline(model_path=str(args.model_path))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"running {len(items)} items through the full pipeline (warmup {args.warmup})…")
    records = run_items(bundle, items, warmup=args.warmup)
    attack_records = [r for r in records if r.split == "attack"]
    clean_records = [r for r in records if r.split == "clean"]
    attack_metrics = compute_metrics(attack_records)
    clean_metrics = compute_metrics(clean_records)

    versions = _versions()
    results: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "run_started_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds"),
        "reproducibility": {
            **versions,
            **_git_fingerprint(),
            "model": _model_fingerprint(args.model_path),
            "manifest_sha256": sha256_file(args.manifest),
            "seed": SEED,
            "run_finished_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"),
            "config": bundle.config_snapshot,
            "latency_note": LATENCY_NOTE,
            "canary_note": CANARY_NOTE,
        },
        "eval_set": {
            "manifest_path": str(args.manifest.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(args.manifest),
            "manifest_frozen_at": manifest.frozen_at,
            "seed": manifest.seed,
            "selection_rule": manifest.selection_rule,
            "sources": {
                name: spec.to_dict() for name, spec in manifest.sources.items()
            },
            "harmbench_note": (
                "HarmBench was considered and skipped: the dataset is gated on "
                "HuggingFace (not anonymously downloadable), which would break "
                "reproducibility from a clean checkout."
            ),
            "counts": {
                "attack": len(attack_records),
                "clean": len(clean_records),
            },
        },
        "attack_metrics": attack_metrics,
        "clean_metrics": clean_metrics,
        "attack_by_source": by_source_metrics(attack_records),
        "clean_by_source": by_source_metrics(clean_records),
        # Exact overall percentiles over all records (not per-split averages).
        "latency_overall_ms": _overall_latency(records),
        "stage_latency_overall_ms": {
            stage: _overall_stage_latency(records, stage)
            for stage in ("normalization", "fast_path", "deep_path")
        },
        "per_item": [record.to_dict() for record in records],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(results), encoding="utf-8"
    )

    print()
    print(
        f"attacks:  n={attack_metrics['n']} blocked={attack_metrics['blocked']} "
        f"(TPR {attack_metrics['block_rate']:.1%}) warned={attack_metrics['warned']} "
        f"(warn {attack_metrics['warn_rate']:.1%})"
    )
    print(
        f"clean:    n={clean_metrics['n']} blocked={clean_metrics['blocked']} "
        f"(FPR {clean_metrics['block_rate']:.1%}) warned={clean_metrics['warned']} "
        f"(warn {clean_metrics['warn_rate']:.1%})"
    )
    print(
        f"latency:  p50={results['latency_overall_ms']['p50']:.1f}ms "
        f"p95={results['latency_overall_ms']['p95']:.1f}ms "
        f"p99={results['latency_overall_ms']['p99']:.1f}ms"
    )
    print(f"results:  {args.out.relative_to(REPO_ROOT)}")
    print(f"markdown: {args.markdown.relative_to(REPO_ROOT)}")
    print(f"total wall time: {time.perf_counter() - started:.1f}s")
    return 0


def _overall_latency(records) -> dict[str, float]:
    from benchmarks.runner import percentile

    values = [r.total_ms for r in records]
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "mean": round(sum(values) / max(len(values), 1), 2),
        "max": round(max(values, default=0.0), 2),
    }


def _overall_stage_latency(records, stage: str) -> dict[str, float]:
    from benchmarks.runner import percentile

    attr = {
        "normalization": "normalization_ms",
        "fast_path": "fast_ms",
        "deep_path": "deep_ms",
    }[stage]
    values = [getattr(r, attr) for r in records if getattr(r, attr) is not None]
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "mean": round(sum(values) / max(len(values), 1), 2),
        "max": round(max(values, default=0.0), 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selfcheck:
        return run_selfcheck(args)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
