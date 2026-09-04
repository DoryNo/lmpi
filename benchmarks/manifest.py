"""Frozen eval-set manifest loading + validation.

Pure stdlib: no network, no HuggingFace imports, no dataset text ever passes
through this module. The manifest records *where* each eval item lives
(dataset repo, split, row index) so the runner can resolve texts at run time
into the gitignored cache directory.

Safety rule (Agent 7 spec): attack prompt texts are never committed to the
repository. :func:`load_manifest` enforces this by rejecting any item whose
serialized form carries known text-bearing keys (``text`` / ``prompt`` /
``content`` / ``messages``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

TRICKY_SOURCE = "tricky_benign"
TRICKY_FILE = "tricky_benign.jsonl"

# Item keys allowed in the manifest JSON. Anything carrying eval *text* is
# deliberately absent — the manifest must stay text-free.
ALLOWED_ITEM_KEYS = frozenset({"id", "source", "row_index", "tags", "text_ref"})
FORBIDDEN_ITEM_KEYS = frozenset({"text", "prompt", "content", "messages"})

ALLOWED_SOURCE_KEYS = frozenset(
    {"repo", "config", "split", "revision", "text_field", "streaming", "description"}
)


@dataclass(frozen=True)
class SourceSpec:
    """One HuggingFace dataset source referenced by the manifest."""

    name: str
    repo: str
    config: str | None
    split: str
    revision: str
    text_field: str
    streaming: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "text_field": self.text_field,
            "streaming": self.streaming,
            "description": self.description,
        }


@dataclass(frozen=True)
class ManifestItem:
    """One eval item descriptor — identifiers only, never text."""

    id: str
    source: str
    label: str  # "attack" | "clean"
    row_index: int | None = None
    tags: dict[str, str] = field(default_factory=dict)
    text_ref: str | None = None


@dataclass(frozen=True)
class EvalManifest:
    """Validated frozen eval manifest."""

    path: Path
    version: int
    frozen_at: str
    seed: int
    selection_rule: str
    sources: dict[str, SourceSpec]
    attacks: tuple[ManifestItem, ...]
    clean: tuple[ManifestItem, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {"attack": len(self.attacks), "clean": len(self.clean)}

    def sources_used(self) -> tuple[SourceSpec, ...]:
        used = {item.source for item in (*self.attacks, *self.clean)}
        return tuple(self.sources[name] for name in sorted(used) if name in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.version,
            "frozen_at": self.frozen_at,
            "seed": self.seed,
            "selection_rule": self.selection_rule,
            "sources": {name: spec.to_dict() for name, spec in self.sources.items()},
            "attack": [_item_to_dict(item) for item in self.attacks],
            "clean": [_item_to_dict(item) for item in self.clean],
        }


def _item_from_dict(raw: dict[str, Any], label: str) -> ManifestItem:
    forbidden = FORBIDDEN_ITEM_KEYS & set(raw)
    if forbidden:
        raise ValueError(
            f"text-bearing keys are not allowed in the manifest (found "
            f"{sorted(forbidden)} in {label} item {raw.get('id')!r}); commit "
            f"IDs and row indices only"
        )
    unknown = set(raw) - ALLOWED_ITEM_KEYS
    if unknown:
        raise ValueError(f"unexpected keys in {label} item: {sorted(unknown)}")
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"{label} item is missing a string id: {raw!r}")
    source = raw.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{label} item {item_id!r} is missing a string source")
    row_index = raw.get("row_index")
    if row_index is not None and (not isinstance(row_index, int) or row_index < 0):
        raise ValueError(f"{label} item {item_id!r} has invalid row_index: {row_index!r}")
    tags = raw.get("tags")
    if tags is not None and not isinstance(tags, dict):
        raise ValueError(f"{label} item {item_id!r} has non-dict tags")
    text_ref = raw.get("text_ref")
    if text_ref is not None and not isinstance(text_ref, str):
        raise ValueError(f"{label} item {item_id!r} has invalid text_ref")
    return ManifestItem(
        id=item_id,
        source=source,
        label=label,
        row_index=row_index,
        tags=dict(tags) if tags else {},
        text_ref=text_ref,
    )


def _item_to_dict(item: ManifestItem) -> dict[str, Any]:
    out: dict[str, Any] = {"id": item.id, "source": item.source}
    if item.row_index is not None:
        out["row_index"] = item.row_index
    if item.tags:
        out["tags"] = item.tags
    if item.text_ref is not None:
        out["text_ref"] = item.text_ref
    return out


def _parse_source(name: str, raw: Any) -> SourceSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"source {name!r} must be a mapping")
    unknown = set(raw) - ALLOWED_SOURCE_KEYS
    if unknown:
        raise ValueError(f"unexpected keys in source {name!r}: {sorted(unknown)}")
    for required in ("repo", "split", "revision", "text_field"):
        value = raw.get(required)
        if not isinstance(value, str) or not value:
            raise ValueError(f"source {name!r} is missing string {required!r}")
    config = raw.get("config")
    if config is not None and not isinstance(config, str):
        raise ValueError(f"source {name!r} config must be a string or null")
    return SourceSpec(
        name=name,
        repo=raw["repo"],
        config=config,
        split=raw["split"],
        revision=raw["revision"],
        text_field=raw["text_field"],
        streaming=bool(raw.get("streaming", False)),
        description=str(raw.get("description", "")),
    )


def load_manifest(path: str | Path) -> EvalManifest:
    """Load and validate a frozen manifest file. Raises ``ValueError`` when
    the file breaks any manifest invariant (schema, uniqueness, text-free)."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"eval manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")

    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest_version {version!r} (expected {MANIFEST_VERSION})"
        )

    sources_raw = data.get("sources")
    if not isinstance(sources_raw, dict) or not sources_raw:
        raise ValueError("manifest must define a non-empty 'sources' mapping")
    sources = {name: _parse_source(name, raw) for name, raw in sources_raw.items()}

    attacks_raw = data.get("attack")
    clean_raw = data.get("clean")
    if not isinstance(attacks_raw, list) or not attacks_raw:
        raise ValueError("manifest must contain a non-empty 'attack' list")
    if not isinstance(clean_raw, list) or not clean_raw:
        raise ValueError("manifest must contain a non-empty 'clean' list")

    attacks = tuple(_item_from_dict(raw, "attack") for raw in attacks_raw)
    clean = tuple(_item_from_dict(raw, "clean") for raw in clean_raw)

    known_sources = {*sources, TRICKY_SOURCE}
    seen: set[str] = set()
    for item in (*attacks, *clean):
        if item.source not in known_sources:
            raise ValueError(
                f"item {item.id!r} references unknown source {item.source!r}"
            )
        if item.source != TRICKY_SOURCE and item.row_index is None:
            raise ValueError(
                f"item {item.id!r} (source {item.source!r}) requires a row_index"
            )
        if item.source == TRICKY_SOURCE and item.row_index is not None:
            raise ValueError(f"tricky item {item.id!r} must not carry a row_index")
        if item.id in seen:
            raise ValueError(f"duplicate item id in manifest: {item.id!r}")
        seen.add(item.id)

    seed = data.get("seed")
    if not isinstance(seed, int):
        raise ValueError("manifest must record an integer 'seed'")

    return EvalManifest(
        path=manifest_path,
        version=MANIFEST_VERSION,
        frozen_at=str(data.get("frozen_at", "")),
        seed=seed,
        selection_rule=str(data.get("selection_rule", "")),
        sources=sources,
        attacks=attacks,
        clean=clean,
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file (used for model + manifest fingerprints)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
