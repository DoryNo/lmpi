"""Resolve frozen manifest items to dataset texts (network + local cache).

The manifest stores dataset coordinates only; this module downloads the
datasets (HuggingFace ``datasets`` library) at run time and caches everything
under the gitignored ``benchmarks/.cache/`` directory. Prompt texts are
returned to the caller but never logged, never printed, and never written
anywhere outside the process.

Call :func:`configure_hf_caches` **before** the first ``datasets`` import
(the CLI does this at startup) so downloads land in the benchmark cache
instead of the user-level HF cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .manifest import (
    EvalManifest,
    ManifestItem,
    SourceSpec,
    TRICKY_FILE,
    TRICKY_SOURCE,
)


def configure_hf_caches(cache_dir: str | Path) -> Path:
    """Point the HF datasets + hub caches at ``benchmarks/.cache``.

    Must run before the ``datasets``/``huggingface_hub`` libraries are first
    imported. Returns the resolved cache directory.
    """
    cache = Path(cache_dir)
    (cache / "datasets").mkdir(parents=True, exist_ok=True)
    (cache / "hub").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    return cache


def load_tricky_benign(tricky_path: Path) -> dict[str, str]:
    """Read the committed hand-written tricky-benign file as ``id → text``."""
    if not tricky_path.is_file():
        raise FileNotFoundError(f"tricky-benign file not found: {tricky_path}")
    items: dict[str, str] = {}
    for line_number, line in enumerate(
        tricky_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        item_id = record.get("id")
        text = record.get("text")
        if not isinstance(item_id, str) or not isinstance(text, str) or not text:
            raise ValueError(
                f"{tricky_path}:{line_number} must define string 'id' and 'text'"
            )
        if item_id in items:
            raise ValueError(f"{tricky_path}:{line_number} duplicate id {item_id!r}")
        items[item_id] = text
    return items


def _streamed_row(dataset: Iterable[dict[str, Any]], row_index: int) -> dict[str, Any]:
    """Fetch one row from a streaming dataset by global index (early stop)."""
    for index, row in enumerate(dataset):
        if index == row_index:
            return row
    raise IndexError(
        f"row_index {row_index} beyond end of streamed split ({index + 1} rows seen)"
    )


def resolve_items(
    manifest: EvalManifest,
    items: list[ManifestItem],
    tricky_path: Path,
    *,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Resolve a list of manifest items to their texts.

    Downloads each referenced dataset once (pinned revision from the
    manifest), picks the needed rows, and returns ``id → text``. Streaming
    sources are iterated once and early-stopped after the highest wanted
    index. Only the configured ``text_field`` of each row is read.
    """
    if cache_dir is not None:
        configure_hf_caches(cache_dir)

    texts: dict[str, str] = {}
    tricky_ids = [item.id for item in items if item.source == TRICKY_SOURCE]
    if tricky_ids:
        tricky = load_tricky_benign(tricky_path)
        for item_id in tricky_ids:
            if item_id not in tricky:
                raise KeyError(
                    f"tricky item {item_id!r} not found in {tricky_path}"
                )
            texts[item_id] = tricky[item_id]

    grouped: dict[str, list[ManifestItem]] = {}
    for item in items:
        if item.source != TRICKY_SOURCE:
            grouped.setdefault(item.source, []).append(item)

    for source_name, source_items in grouped.items():
        spec = manifest.sources.get(source_name)
        if spec is None:
            raise KeyError(f"item references unknown source {source_name!r}")
        rows = _load_rows(spec, {item.row_index for item in source_items if item.row_index is not None})
        for item in source_items:
            row = rows.get(item.row_index)  # type: ignore[arg-type]
            if row is None:
                raise IndexError(
                    f"{source_name}: row {item.row_index} not found in resolved rows"
                )
            text = row.get(spec.text_field) if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{source_name} row {item.row_index}: field {spec.text_field!r} "
                    f"is not a non-empty string"
                )
            texts[item.id] = text
    return texts


def _load_rows(
    spec: SourceSpec, wanted_indices: set[int]
) -> dict[int, dict[str, Any]]:
    """Load one dataset split (pinned revision) and return wanted rows."""
    from datasets import load_dataset

    dataset = load_dataset(
        spec.repo,
        spec.config,
        split=spec.split,
        revision=spec.revision,
        streaming=spec.streaming,
    )
    rows: dict[int, dict[str, Any]] = {}
    if spec.streaming:
        max_index = max(wanted_indices)
        for index, row in enumerate(dataset):
            if index in wanted_indices:
                rows[index] = dict(row)
            if index >= max_index:
                break
    else:
        for index in sorted(wanted_indices):
            rows[index] = dict(dataset[index])
    missing = wanted_indices - set(rows)
    if missing:
        raise IndexError(f"{spec.repo}: rows {sorted(missing)} not found")
    return rows
