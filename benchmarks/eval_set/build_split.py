#!/usr/bin/env python
"""Build the v1.1 tuning/held-out split of the frozen eval set.

Deterministic, stratified 60/40 partition of the frozen manifest used for
the documented threshold-tuning round (ROADMAP.md v1.1):

- **Tuning** (60%): everything a tuning decision is allowed to look at —
  failure analysis, pattern additions, threshold sweeps.
- **Held-out** (40%): touched exactly once, by the final full-set run; the
  reported v1.1 metrics come from here, never from the tuning subset.

The split is stratified by (label, source) so every stratum keeps its
natural proportion. It is an *extension* of the frozen manifest: item
membership is fully determined by (SPLIT_SEED, stratum sizes) and is
re-derivable; no text is involved or stored.

Usage::

    python benchmarks/eval_set/build_split.py            # writes split.json
    python benchmarks/eval_set/build_split.py --verify   # re-derive + compare
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest import load_manifest  # noqa: E402

SPLIT_SCHEMA = "lmpi-eval-split/1"

# Dedicated seed, independent from the eval-set selection seeds (20260905 /
# 20260906). Recorded in split.json so the partition is reproducible.
SPLIT_SEED = 20260911

TUNING_FRACTION = 0.6

# Per-stratum tuning sizes: round(0.6 * stratum size) evaluated by hand:
#   jbb attacks 100 -> 60, wild attacks 100 -> 60,
#   jbb benign 100 -> 60, ultrachat 170 -> 102, tricky 30 -> 18.
EXPECTED_STRATA = {
    ("attack", "jbb"): 60,
    ("attack", "wild"): 60,
    ("clean", "jbb"): 60,
    ("clean", "ultrachat"): 102,
    ("clean", "tricky_benign"): 18,
}

SOURCE_KEY = {
    "jbb": "jbb",
    "wild": "wild",
    "ultrachat": "ultrachat",
    "tricky_benign": "tricky_benign",
}


def stratum_of(item_label: str, item_source: str) -> tuple[str, str]:
    """Map a manifest item onto its (label, tuning-source-key) stratum."""
    return (item_label, SOURCE_KEY[item_source])


def build_split(manifest_path: Path) -> dict:
    """Derive the split from the manifest + SPLIT_SEED (pure, no I/O input)."""
    manifest = load_manifest(str(manifest_path))
    strata: dict[tuple[str, str], list[str]] = {}
    for item in manifest.attacks:
        strata.setdefault(("attack", SOURCE_KEY[item.source]), []).append(item.id)
    for item in manifest.clean:
        strata.setdefault(("clean", SOURCE_KEY[item.source]), []).append(item.id)

    # Fixed stratum iteration order -> identical rng consumption everywhere.
    for key in sorted(strata):
        expected = EXPECTED_STRATA.get(key)
        size = round(TUNING_FRACTION * len(strata[key]))
        if expected is not None and size != expected:
            raise RuntimeError(
                f"stratum {key}: derived tuning size {size} != expected "
                f"{expected} (manifest changed? split must be rebuilt "
                f"consciously)"
            )

    tuning: dict[str, list[str]] = {"attack": [], "clean": []}
    held_out: dict[str, list[str]] = {"attack": [], "clean": []}
    for (label, source) in sorted(strata):
        ids = list(strata[(label, source)])  # manifest order (stable)
        rng = random.Random(f"{SPLIT_SEED}:{label}:{source}")
        rng.shuffle(ids)
        k = round(TUNING_FRACTION * len(ids))
        picked = ids[:k]
        rest = ids[k:]
        tuning[label].extend(picked)
        held_out[label].extend(rest)

    return {
        "schema": SPLIT_SCHEMA,
        "seed": SPLIT_SEED,
        "tuning_fraction": TUNING_FRACTION,
        "method": (
            "stratified by (label, source); per stratum: items in manifest "
            "order, random.Random(f'{seed}:{label}:{source}').shuffle, first "
            "round(0.6 * n) -> tuning, rest -> held_out; id lists sorted"
        ),
        "manifest_selection_seeds": {"eval_seed": manifest.seed},
        "tuning": {label: sorted(ids) for label, ids in tuning.items()},
        "held_out": {label: sorted(ids) for label, ids in held_out.items()},
        "counts": {
            "tuning": {
                "attack": len(tuning["attack"]),
                "clean": len(tuning["clean"]),
            },
            "held_out": {
                "attack": len(held_out["attack"]),
                "clean": len(held_out["clean"]),
            },
        },
    }


def verify_split(split: dict, manifest_path: Path) -> list[str]:
    """Re-derive and compare; also check a clean partition of the manifest."""
    manifest = load_manifest(str(manifest_path))
    problems: list[str] = []
    all_ids = {item.id for item in (*manifest.attacks, *manifest.clean)}
    tuning_ids = {i for label in ("attack", "clean") for i in split["tuning"][label]}
    held_ids = {i for label in ("attack", "clean") for i in split["held_out"][label]}
    if tuning_ids & held_ids:
        problems.append(f"overlap: {sorted(tuning_ids & held_ids)[:5]}…")
    if tuning_ids | held_ids != all_ids:
        problems.append(
            f"union != manifest (missing {sorted(all_ids - (tuning_ids | held_ids))[:5]}…)"
        )
    rebuilt = build_split(manifest_path)
    for label in ("attack", "clean"):
        for subset in ("tuning", "held_out"):
            if rebuilt[subset][label] != split[subset][label]:
                problems.append(f"{subset}/{label} differs from re-derivation")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "split.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive the committed split and check it still matches",
    )
    args = parser.parse_args(argv)

    if args.verify:
        split = json.loads(args.out.read_text(encoding="utf-8"))
        problems = verify_split(split, args.manifest)
        if problems:
            print("VERIFY FAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(
            "verify OK: split.json matches its seed/derivation and partitions "
            "the manifest exactly"
        )
        return 0

    split = build_split(args.manifest)
    args.out.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} (seed {SPLIT_SEED})")
    print(f"  tuning:   {split['counts']['tuning']}")
    print(f"  held_out: {split['counts']['held_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
