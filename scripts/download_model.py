#!/usr/bin/env python
"""Download the deep-path ONNX model + tokenizer into models/ (gitignored).

Usage::

    python scripts/download_model.py
    python scripts/download_model.py --repo protectai/deberta-v3-base-prompt-injection-v2 --out models/deberta-v3-base-prompt-injection-v2

Downloads, from the ``onnx/`` folder of the HuggingFace repo:

- ``model_quantized.onnx`` when available, else ``model.onnx`` (the v2
  repo currently ships **no** quantized variant — the full-precision file
  is used and the fact is printed here and noted in the README);
- ``tokenizer.json`` (+ tokenizer config files) for the ``tokenizers``
  library loader.

Model binaries are never committed (``models/`` is gitignored). After the
download, enable the stage with ``LMPI_DEEP_PATH_ENABLED=true``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_OUT = Path("models") / "deberta-v3-base-prompt-injection-v2"

QUANTIZED_FILENAME = "model_quantized.onnx"
PLAIN_FILENAME = "model.onnx"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
    "spm.model",
)

# onnx/ prefix is flattened away in the output dir, matching the layout
# OnnxRuntimeBackend expects.
ONNX_PREFIX = "onnx/"


def _download(repo_id: str, filename: str, out_dir: Path) -> Path:
    """Download one repo file into ``out_dir`` (flat) and return its path."""
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=ONNX_PREFIX + filename, local_dir=out_dir)
    )
    target = out_dir / filename
    if downloaded != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(target))
    return target


def _print_file(label: str, path: Path) -> None:
    print(f"  {label:<28} {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace repo id")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading deep-path model from {args.repo} into {out_dir}/")
    try:
        try:
            from huggingface_hub import EntryNotFoundError
        except ImportError:  # older hub versions
            from huggingface_hub.utils import EntryNotFoundError
        try:
            model_path = _download(args.repo, QUANTIZED_FILENAME, out_dir)
            quantized = True
        except EntryNotFoundError:
            # The v2 repo publishes no quantized ONNX variant; fall back to
            # the full-precision export instead of failing outright.
            model_path = _download(args.repo, PLAIN_FILENAME, out_dir)
            quantized = False
        for filename in TOKENIZER_FILES:
            _download(args.repo, filename, out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        print(
            "The deep path stays disabled; the proxy keeps running without it.",
            file=sys.stderr,
        )
        return 1

    print()
    _print_file("model (quantized)" if quantized else "model (NOT quantized)", model_path)
    for filename in TOKENIZER_FILES:
        path = out_dir / filename
        if path.is_file():
            _print_file(filename, path)
    print()
    if not quantized:
        print(
            "NOTE: no quantized variant is published for this repo; "
            "the full-precision ONNX export is used instead (see README)."
        )
    print("Done. Enable the stage with LMPI_DEEP_PATH_ENABLED=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
