"""Offline unit tests for the frozen eval manifest (benchmarks.manifest)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.manifest import (
    TRICKY_FILE,
    TRICKY_SOURCE,
    load_manifest,
    sha256_file,
    sha256_text,
)

TRICKY_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "eval_set" / TRICKY_FILE


def _manifest_path(tmp_path: Path, **overrides) -> Path:
    manifest = {
        "manifest_version": 1,
        "frozen_at": "2026-09-04T00:00:00+00:00",
        "seed": 1,
        "selection_rule": "test fixture",
        "sources": {
            "stub": {
                "repo": "org/stub",
                "config": None,
                "split": "train",
                "revision": "deadbeef",
                "text_field": "prompt",
                "streaming": False,
                "description": "fixture source",
            }
        },
        "attack": [{"id": "stub-a-000", "source": "stub", "row_index": 0}],
        "clean": [
            {"id": "tricky-001", "source": TRICKY_SOURCE},
            {"id": "stub-c-000", "source": "stub", "row_index": 1},
        ],
    }
    manifest.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_manifest_ok(tmp_path: Path) -> None:
    manifest = load_manifest(_manifest_path(tmp_path))
    assert manifest.version == 1
    assert manifest.seed == 1
    assert manifest.counts == {"attack": 1, "clean": 2}
    assert [item.id for item in manifest.attacks] == ["stub-a-000"]
    assert manifest.sources_used()[0].repo == "org/stub"


@pytest.mark.parametrize(
    "key", ["text", "prompt", "content", "messages"]
)
def test_rejects_text_bearing_keys(tmp_path: Path, key: str) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["attack"][0][key] = "do not commit attack texts"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="text-bearing keys"):
        load_manifest(path)


def test_rejects_unknown_source(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["attack"][0]["source"] = "mystery"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source"):
        load_manifest(path)


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clean"][0]["id"] = "stub-a-000"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate item id"):
        load_manifest(path)


def test_rejects_unknown_item_keys(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["attack"][0]["gold"] = "42"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_manifest(path)


def test_row_index_required_for_dataset_sources(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["attack"][0]["row_index"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="row_index"):
        load_manifest(path)


def test_tricky_item_must_not_carry_row_index(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clean"][0]["row_index"] = 5
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must not carry a row_index"):
        load_manifest(path)


def test_committed_tricky_benign_file_is_structurally_valid() -> None:
    from benchmarks.hf_sources import load_tricky_benign

    items = load_tricky_benign(TRICKY_PATH)
    assert len(items) == 30
    expected = {f"tricky-{index:03d}" for index in range(1, 31)}
    assert set(items) == expected
    assert all(text.strip() for text in items.values())


def test_load_tricky_benign_rejects_bad_rows(tmp_path: Path) -> None:
    from benchmarks.hf_sources import load_tricky_benign

    path = tmp_path / "tricky.jsonl"
    path.write_text('{"id": "tricky-001", "text": "ok"}\n', encoding="utf-8")
    assert load_tricky_benign(path) == {"tricky-001": "ok"}

    path.write_text('{"id": "tricky-001", "text": "ok"}\n'
                    '{"id": "tricky-001", "text": "dup"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_tricky_benign(path)

    path.write_text('{"id": "tricky-001"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="string 'id' and 'text'"):
        load_tricky_benign(path)


def test_sha256_helpers(tmp_path: Path) -> None:
    payload = tmp_path / "blob.bin"
    payload.write_bytes(b"lmpi benchmark")
    assert sha256_file(payload) == hashlib.sha256(b"lmpi benchmark").hexdigest()
    assert sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()
