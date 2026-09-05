"""Tuning/held-out split loading + validation (v1.1 tuning round).

``split.json`` is a committed *extension* of the frozen manifest: it
partitions the 500 eval items into a tuning subset (60%, the only subset a
tuning decision may look at) and a held-out subset (40%, touched exactly
once by the final benchmark run — reported metrics come from here).

Loader is pure stdlib and text-free: it validates that the two subsets are a
clean partition of the manifest IDs and that the committed file re-derives
from its recorded seed (see ``benchmarks/eval_set/build_split.py``).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import EvalManifest, load_manifest

SPLIT_SCHEMA = "lmpi-eval-split/1"
SPLIT_FILE = "split.json"
TUNING = "tuning"
HELD_OUT = "held_out"
SPLIT_SEED = 20260911
TUNING_FRACTION = 0.6

# Mirrors build_split.EXPECTED_STRATA — kept in sync by the verify step.
_SOURCE_KEY = {"jbb": "jbb", "wild": "wild", "ultrachat": "ultrachat", "tricky_benign": "tricky_benign"}


@dataclass(frozen=True)
class EvalSplit:
    """Validated tuning/held-out partition (ID sets only, never text)."""

    path: Path
    seed: int
    tuning: dict[str, frozenset[str]]
    held_out: dict[str, frozenset[str]]

    def subset_of(self, subset: str, label: str) -> frozenset[str]:
        """IDs of one (subset, label) cell: e.g. ``('tuning', 'attack')``."""
        table = {TUNING: self.tuning, HELD_OUT: self.held_out}
        if subset not in table:
            raise KeyError(f"unknown subset {subset!r}")
        return table[subset][label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "counts": {
                TUNING: {label: len(ids) for label, ids in self.tuning.items()},
                HELD_OUT: {label: len(ids) for label, ids in self.held_out.items()},
            },
        }


def _derive(manifest: EvalManifest) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Re-derive the split from the manifest + SPLIT_SEED (build_split logic)."""
    strata: dict[tuple[str, str], list[str]] = {}
    for item in manifest.attacks:
        strata.setdefault(("attack", _SOURCE_KEY[item.source]), []).append(item.id)
    for item in manifest.clean:
        strata.setdefault(("clean", _SOURCE_KEY[item.source]), []).append(item.id)

    tuning: dict[str, list[str]] = {"attack": [], "clean": []}
    held_out: dict[str, list[str]] = {"attack": [], "clean": []}
    for key in sorted(strata):
        label, source = key
        ids = list(strata[key])
        rng = random.Random(f"{SPLIT_SEED}:{label}:{source}")
        rng.shuffle(ids)
        k = round(TUNING_FRACTION * len(ids))
        tuning[label].extend(ids[:k])
        held_out[label].extend(ids[k:])
    return tuning, held_out


def load_split(
    path: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    verify: bool = True,
) -> EvalSplit:
    """Load + validate the committed split against the frozen manifest."""
    root = Path(__file__).resolve().parent / "eval_set"
    split_path = Path(path) if path is not None else root / SPLIT_FILE
    manifest_file = Path(manifest_path) if manifest_path is not None else root / "manifest.json"

    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if raw.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"{split_path}: unexpected schema {raw.get('schema')!r}")

    manifest = load_manifest(str(manifest_file))
    all_ids = {item.id for item in (*manifest.attacks, *manifest.clean)}
    tuning_raw = raw.get(TUNING, {})
    held_raw = raw.get(HELD_OUT, {})
    for label in ("attack", "clean"):
        for subset_raw in (tuning_raw, held_raw):
            ids = subset_raw.get(label, [])
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError(f"{split_path}: {label} ids must be a list of strings")
    unknown = {
        i
        for label in ("attack", "clean")
        for ids in (tuning_raw[label], held_raw[label])
        for i in ids
    } - all_ids
    if unknown:
        raise ValueError(f"{split_path}: ids not in the manifest: {sorted(unknown)[:5]}…")
    tuning_ids = {i for label in ("attack", "clean") for i in tuning_raw[label]}
    held_ids = {i for label in ("attack", "clean") for i in held_raw[label]}
    if tuning_ids & held_ids:
        raise ValueError(f"{split_path}: tuning/held-out overlap {sorted(tuning_ids & held_ids)[:5]}…")
    if tuning_ids | held_ids != all_ids:
        raise ValueError(f"{split_path}: split does not cover the manifest exactly")

    split = EvalSplit(
        path=split_path,
        seed=int(raw["seed"]),
        tuning={label: frozenset(tuning_raw[label]) for label in ("attack", "clean")},
        held_out={label: frozenset(held_raw[label]) for label in ("attack", "clean")},
    )
    if verify and int(raw["seed"]) == SPLIT_SEED:
        derived_tuning, derived_held = _derive(manifest)
        if (
            {i for ids in derived_tuning.values() for i in ids} != tuning_ids
            or {i for ids in derived_held.values() for i in ids} != held_ids
        ):
            raise ValueError(
                f"{split_path}: membership does not re-derive from seed "
                f"{SPLIT_SEED} — was the split edited by hand?"
            )
    return split
