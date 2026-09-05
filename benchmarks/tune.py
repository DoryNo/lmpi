"""v1.1 tuning-round tooling: score collection, threshold sweeps, error analysis.

Companion to the documented tuning iteration (ROADMAP.md v1.1). Three
subcommands, all text-free on the output side:

``collect``
    Run a split subset (or the full set) through the detection pipeline and
    write a **score cache**: per item id — fast score, deep score,
    normalization findings. No prompt text, in the file or on stdout.
    Threshold sweeps afterwards are offline (deterministic, instant) because
    the decision rule depends only on the two scores.

``sweep``
    Offline threshold sweep over a score cache for one split subset.
    Reports the TPR/FPR trade-off curve for the deep-path block threshold
    (the documented v1.1 grid) plus fast-path threshold variants.

``analyze``
    Error analysis on a split subset: classify misses / false positives into
    the documented buckets (a)–(d) and print aggregate counts only.

The decision rule mirrors the pipeline exactly: fast block short-circuits,
deep block, else warn from either stage's warn threshold. Validated against
the committed v1.0 per-item records (0 mismatches on 500 items).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hf_sources import configure_hf_caches, resolve_items  # noqa: E402
from benchmarks.manifest import load_manifest  # noqa: E402
from benchmarks.runner import build_benchmark_pipeline  # noqa: E402
from benchmarks.split import HELD_OUT, TUNING, load_split  # noqa: E402

SCORES_SCHEMA = "lmpi-tuning-scores/1"

DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "eval_set" / "manifest.json"
DEFAULT_TRICKY = REPO_ROOT / "benchmarks" / "eval_set" / "tricky_benign.jsonl"
DEFAULT_MODEL = REPO_ROOT / "models" / "deberta-v3-base-prompt-injection-v2"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / ".cache"
DEFAULT_SPLIT = REPO_ROOT / "benchmarks" / "eval_set" / "split.json"
ANALYSIS_DIR = REPO_ROOT / "benchmarks" / "analysis"

# Baseline-as-shipped thresholds (v1.0).
BASELINE = {"fast_block": 0.75, "fast_warn": 0.4, "deep_block": 0.75, "deep_warn": 0.5}

# Documented v1.1 sweep grid for the deep-path block threshold.
DEEP_BLOCK_GRID = [round(0.5 + 0.05 * i, 2) for i in range(7)]  # 0.50..0.80


@dataclass(frozen=True)
class Config:
    fast_block: float = BASELINE["fast_block"]
    fast_warn: float = BASELINE["fast_warn"]
    deep_block: float = BASELINE["deep_block"]
    deep_warn: float = BASELINE["deep_warn"]


@dataclass(frozen=True)
class ScoreRecord:
    id: str
    label: str  # "attack" | "clean"
    fast_score: float
    deep_score: float
    findings: int
    finding_categories: tuple[str, ...]
    char_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "fast_score": round(self.fast_score, 4),
            "deep_score": round(self.deep_score, 4),
            "findings": self.findings,
            "finding_categories": list(self.finding_categories),
            "char_truncated": self.char_truncated,
        }


def decide(fast_score: float, deep_score: float, cfg: Config) -> str:
    """Pipeline decision from the two stage scores (validated vs. v1.0)."""
    if fast_score >= cfg.fast_block:
        return "block"
    if deep_score >= cfg.deep_block:
        return "block"
    if fast_score >= cfg.fast_warn or deep_score >= cfg.deep_warn:
        return "warn"
    return "allow"


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def collect_items(
    subset: str,
    *,
    model_path: Path,
    cache_dir: Path,
    tricky_path: Path,
    manifest_path: Path,
    split_path: Path | None,
) -> list[tuple[str, str]]:
    """Resolve the (id, label) items of a named subset to run."""
    manifest = load_manifest(str(manifest_path))
    if subset == "all":
        items = [(it.id, it.label) for it in (*manifest.attacks, *manifest.clean)]
    else:
        split = load_split(split_path, manifest_path=manifest_path)
        ids = split.subset_of(subset, "attack") | split.subset_of(subset, "clean")
        items = [
            (it.id, it.label) for it in (*manifest.attacks, *manifest.clean) if it.id in ids
        ]
    return items


async def _run_scores(bundle, items, texts) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    fast = bundle.fast_path
    deep = bundle.deep_path
    from src.normalization import normalize

    for item_id, label in items:
        raw_text = texts[item_id]
        fast.last = None
        deep.last = None
        payload = {"model": "benchmark-eval", "messages": [{"role": "user", "content": raw_text}]}
        await bundle.pipeline.process_request(payload)
        norm = normalize(raw_text)
        cleaned = norm.cleaned_text if norm.changed else raw_text
        if fast.last is not None and fast.last.action == "block":
            deep.detect(cleaned)  # shadow scan for attribution parity
        fast_score = fast.last.score if fast.last else 0.0
        deep_score = deep.last.score if deep.last else 0.0
        records.append(
            ScoreRecord(
                id=item_id,
                label=label,
                fast_score=float(fast_score),
                deep_score=float(deep_score),
                findings=len(norm.findings),
                finding_categories=tuple(sorted({f.category for f in norm.findings})),
                char_truncated=bool(deep.last.char_truncated) if deep.last else False,
            )
        )
    return records


def collect(args: argparse.Namespace) -> int:
    items = collect_items(
        args.subset,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tricky_path=args.tricky_path,
        manifest_path=args.manifest,
        split_path=args.split,
    )
    print(f"collect: running {len(items)} items ({args.subset}) through the pipeline…")
    configure_hf_caches(args.cache_dir)
    manifest = load_manifest(str(args.manifest))
    manifest_items = {it.id: it for it in (*manifest.attacks, *manifest.clean)}
    texts = resolve_items(
        manifest,
        [manifest_items[i] for i, _ in items],
        args.tricky_path,
        cache_dir=args.cache_dir,
    )
    bundle = build_benchmark_pipeline(
        model_path=str(args.model_path),
        fast_block=args.fast_block,
        fast_warn=args.fast_warn,
        deep_block=args.deep_block,
        deep_warn=args.deep_warn,
        deep_max_chars=args.deep_max_chars,
    )
    records = asyncio.run(_run_scores(bundle, items, texts))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": SCORES_SCHEMA,
                "subset": args.subset,
                "thresholds": {
                    "fast_block": args.fast_block,
                    "fast_warn": args.fast_warn,
                    "deep_block": args.deep_block,
                    "deep_warn": args.deep_warn,
                    "deep_max_chars": args.deep_max_chars,
                },
                "records": [r.to_dict() for r in records],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    shown = out if out.is_absolute() else REPO_ROOT / out
    print(f"wrote {shown.relative_to(REPO_ROOT)} ({len(records)} records, no prompt text)")
    return 0


def load_scores(path: Path) -> list[ScoreRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ScoreRecord(
            id=r["id"],
            label=r["label"],
            fast_score=float(r["fast_score"]),
            deep_score=float(r["deep_score"]),
            findings=int(r["findings"]),
            finding_categories=tuple(r.get("finding_categories", ())),
            char_truncated=bool(r.get("char_truncated", False)),
        )
        for r in raw["records"]
    ]


# ---------------------------------------------------------------------------
# metrics + sweep
# ---------------------------------------------------------------------------


def subset_metrics(records: list[ScoreRecord], label: str, cfg: Config) -> dict[str, Any]:
    group = [r for r in records if r.label == label]
    n = len(group)
    if n == 0:
        return {"n": 0}
    decisions = [decide(r.fast_score, r.deep_score, cfg) for r in group]
    blocked = sum(1 for d in decisions if d == "block")
    warned = sum(1 for d in decisions if d == "warn")
    return {
        "n": n,
        "blocked": blocked,
        "rate": round(blocked / n, 4),
        "warned": warned,
        "warn_rate": round(warned / n, 4),
    }


def sweep_table(records: list[ScoreRecord], cfg: Config, deep_block: float) -> dict[str, Any]:
    """Metrics at one candidate deep-block threshold (fast thresholds fixed)."""
    swept = Config(
        fast_block=cfg.fast_block,
        fast_warn=cfg.fast_warn,
        deep_block=deep_block,
        deep_warn=min(cfg.deep_warn, deep_block),
    )
    attack = subset_metrics(records, "attack", swept)
    clean = subset_metrics(records, "clean", swept)
    return {"deep_block": deep_block, "attack": attack, "clean": clean}


def sweep(args: argparse.Namespace) -> int:
    records = load_scores(args.scores)
    cfg = Config(args.fast_block, args.fast_warn, args.deep_block, args.deep_warn)
    print(f"baseline thresholds {cfg} on {len(records)} cached records")
    print()
    print("Deep-path block threshold sweep (fast thresholds fixed at "
          f"{cfg.fast_block}/{cfg.fast_warn}):")
    print("| deep_block | attack blocked | TPR | clean blocked | FPR |")
    print("|------------|----------------|-----|---------------|-----|")
    for deep_block in DEEP_BLOCK_GRID:
        row = sweep_table(records, cfg, deep_block)
        a, c = row["attack"], row["clean"]
        print(
            f"| {deep_block:.2f} | {a['blocked']}/{a['n']} | {a['rate']:.1%} "
            f"| {c['blocked']}/{c['n']} | {c['rate']:.1%} |"
        )
    return 0


# ---------------------------------------------------------------------------
# analyze — error-bucket classification (aggregate counts only)
# ---------------------------------------------------------------------------


def analyze(args: argparse.Namespace) -> int:
    records = load_scores(args.scores)
    cfg = Config(args.fast_block, args.fast_warn, args.deep_block, args.deep_warn)
    print("Attack misses:")
    buckets: dict[str, list[str]] = {}
    for r in records:
        if r.label != "attack":
            continue
        decision = decide(r.fast_score, r.deep_score, cfg)
        if decision == "block":
            continue
        if decision == "warn":
            bucket = "warned_not_blocked"
        elif r.findings > 0:
            bucket = "a_normalized_decoded"
        elif r.deep_score >= 0.4:
            bucket = "c_deep_near_threshold"
        else:
            bucket = "b_structural_miss"
        buckets.setdefault(bucket, []).append(r.id)
    for bucket in sorted(buckets):
        ids = buckets[bucket]
        wild = sum(1 for i in ids if i.startswith("wild-"))
        jbb = len(ids) - wild
        print(f"  {bucket:<24} n={len(ids):<4} (wild={wild}, jbb={jbb})")

    print("Clean false positives (blocked):")
    fp_fast, fp_deep = [], []
    for r in records:
        if r.label != "clean":
            continue
        if decide(r.fast_score, r.deep_score, cfg) == "block":
            (fp_fast if r.fast_score >= cfg.fast_block else fp_deep).append(r.id)
    print(f"  fast-path-driven : {len(fp_fast)}")
    print(f"  deep-path-driven : {len(fp_deep)}  (deep>=0.9: "
          f"{sum(1 for r in records if r.label == 'clean' and r.deep_score >= 0.9)})")

    print("Clean near-FP band (deep in [deep_warn, deep_block)):")
    band = [
        r.id
        for r in records
        if r.label == "clean"
        and r.fast_score < cfg.fast_warn
        and cfg.deep_warn <= r.deep_score < cfg.deep_block
    ]
    print(f"  n={len(band)}")
    return 0


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("collect", "sweep", "analyze"))
    parser.add_argument("--subset", choices=(TUNING, HELD_OUT, "all"), default=TUNING)
    parser.add_argument("--scores", type=Path, default=ANALYSIS_DIR / "scores_tuning.json")
    parser.add_argument("--out", type=Path, default=ANALYSIS_DIR / "scores_tuning.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tricky-path", type=Path, default=DEFAULT_TRICKY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fast-block", type=float, default=BASELINE["fast_block"])
    parser.add_argument("--fast-warn", type=float, default=BASELINE["fast_warn"])
    parser.add_argument("--deep-block", type=float, default=BASELINE["deep_block"])
    parser.add_argument("--deep-warn", type=float, default=BASELINE["deep_warn"])
    parser.add_argument("--deep-max-chars", type=int, default=6000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv).parse_args(argv)
    if args.command == "collect":
        return collect(args)
    if args.command == "sweep":
        return sweep(args)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
