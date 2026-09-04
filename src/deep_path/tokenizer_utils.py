"""Tokenizer loading and truncation helpers for the deep path (stage 3).

Uses the lightweight ``tokenizers`` library (Rust-backed) — NOT
``transformers``/``torch`` — to keep the dependency footprint small. The
ONNX export of ``protectai/deberta-v3-base-prompt-injection-v2`` ships a
``tokenizer.json`` alongside ``model.onnx``; both live in the model
directory produced by ``scripts/download_model.py``.

Truncation model: standard **first-512-token** truncation (the model was
trained with a 512-token window — see ``max_position_embeddings`` in its
config). Trade-off, documented deliberately: a prompt longer than 512
tokens only gets its *beginning* scanned; an injection payload near the
end of a very long prompt can be missed. Latency + training-window
consistency win over clever windowing for v1.
"""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - import guard exercised indirectly
    from tokenizers import Tokenizer
except ImportError:  # onnxruntime-free environments still import the package
    Tokenizer = None  # type: ignore[assignment]

TOKENIZERS_AVAILABLE = Tokenizer is not None

DEFAULT_MAX_LENGTH = 512

TOKENIZER_FILENAME = "tokenizer.json"

_CLS_TOKEN = "[CLS]"
_SEP_TOKEN = "[SEP]"


def require_tokenizers() -> None:
    """Raise an actionable error when the ``tokenizers`` package is missing."""
    if not TOKENIZERS_AVAILABLE:
        raise RuntimeError(
            "The 'tokenizers' package is required for the deep path "
            "(pip install tokenizers)"
        )


def tokenizer_available(path: Path) -> bool:
    """Lazy availability probe: True when tokenizer.json exists on disk."""
    try:
        return Path(path).is_file()
    except OSError:
        return False


def load_tokenizer(
    path: Path | str, *, max_length: int = DEFAULT_MAX_LENGTH
) -> "Tokenizer":
    """Load a ``tokenizer.json`` with truncation to ``max_length`` tokens.

    The exported DeBERTa tokenizer.json carries no post-processor in some
    revisions, so a ``[CLS] ... [SEP]`` template is attached when missing —
    the ONNX graph was trained with those special tokens and mis-scores
    without them.
    """
    require_tokenizers()
    assert Tokenizer is not None  # for type checkers
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {path} — run "
            f"`python scripts/download_model.py` first (see README, Deep Path)"
        )
    tokenizer = Tokenizer.from_file(str(path))

    cls_id = tokenizer.token_to_id(_CLS_TOKEN)
    sep_id = tokenizer.token_to_id(_SEP_TOKEN)
    if cls_id is not None and sep_id is not None:
        from tokenizers.processors import TemplateProcessing

        tokenizer.post_processor = TemplateProcessing(
            single=f"{_CLS_TOKEN} $A {_SEP_TOKEN}",
            pair=f"{_CLS_TOKEN} $A {_SEP_TOKEN} $B {_SEP_TOKEN}",
            special_tokens=[
                (_CLS_TOKEN, cls_id),
                (_SEP_TOKEN, sep_id),
            ],
        )

    # Truncation keeps the FIRST max_length tokens (standard behaviour).
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer
