"""Model backends for the deep path (stage 3).

:class:`OnnxRuntimeBackend` runs the ONNX export of
``protectai/deberta-v3-base-prompt-injection-v2`` (a *pretrained* prompt
injection classifier — NOT fine-tuned by LMPI; see README honesty note)
through ``onnxruntime`` with the lightweight ``tokenizers`` tokenizer.

:class:`StubBackend` is a deterministic in-memory fake so tests never
require the network or the real model binary.

Label mapping (from the model's ``config.json``): index 0 = ``SAFE``
(benign), index 1 = ``INJECTION`` — :meth:`predict` returns
``(benign, injection)`` probability pairs that sum to 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from .tokenizer_utils import DEFAULT_MAX_LENGTH, TOKENIZERS_AVAILABLE, load_tokenizer

MODEL_FILENAME_QUANTIZED = "model_quantized.onnx"
MODEL_FILENAME_PLAIN = "model.onnx"


def softmax_rows(logits: list[list[float]]) -> list[list[float]]:
    """Numerically stable row-wise softmax over plain-python logits.

    Kept dependency-free (no numpy import at module level) so it can be
    unit-tested with pure-python fakes.
    """
    rows: list[list[float]] = []
    for row in logits:
        if not row:
            rows.append([])
            continue
        peak = max(row)
        exps = [math.exp(value - peak) for value in row]
        total = sum(exps)
        rows.append([value / total for value in exps])
    return rows


@runtime_checkable
class ModelBackend(Protocol):
    """Anything that can score texts as ``(benign, injection)`` pairs."""

    model_name: str
    quantized: bool

    def predict(self, texts: list[str]) -> list[tuple[float, float]]:
        """Return one ``(benign_prob, injection_prob)`` pair per input text."""
        ...


class StubBackend:
    """Deterministic fake backend for tests — no onnx, no tokenizer, no I/O.

    Returns the same score pair for every text; records how often it was
    called and what it saw, so pipeline tests can assert short-circuiting
    (fast-path block ⇒ deep path never called).
    """

    def __init__(self, scores: tuple[float, float] = (0.05, 0.95)) -> None:
        self._scores = scores
        self.calls = 0
        self.seen_texts: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def quantized(self) -> bool:
        return False

    def predict(self, texts: list[str]) -> list[tuple[float, float]]:
        self.calls += 1
        self.seen_texts.append(list(texts))
        return [self._scores for _ in texts]


class OnnxRuntimeBackend:
    """ONNX Runtime inference backend with HF-``tokenizers`` preprocessing.

    Loads ``model_quantized.onnx`` when present, else ``model.onnx`` (the
    upstream repo currently ships no quantized variant — README notes the
    fallback). Raises :class:`FileNotFoundError` with actionable guidance
    when the model directory is incomplete; the pipeline then degrades
    gracefully (stage disabled + one-time warning).
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        base = Path(model_path)
        quantized_file = base / MODEL_FILENAME_QUANTIZED
        plain_file = base / MODEL_FILENAME_PLAIN
        if quantized_file.is_file():
            model_file, self.quantized = quantized_file, True
        elif plain_file.is_file():
            model_file, self.quantized = plain_file, False
        else:
            raise FileNotFoundError(
                f"ONNX model not found under {base} (looked for "
                f"{MODEL_FILENAME_QUANTIZED} / {MODEL_FILENAME_PLAIN}) — run "
                f"`python scripts/download_model.py` first (see README)"
            )
        self.model_path = base
        self.model_file = str(model_file)
        self.model_name = base.name
        self.max_length = max_length

        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError(
                "The 'tokenizers' package is required for the deep path "
                "(pip install tokenizers)"
            )
        # Imports below are kept inside __init__ so that merely importing
        # src.deep_path never hard-fails when the heavy deps are absent —
        # the pipeline degrades gracefully instead.
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime (and numpy) are required for the deep path "
                "(pip install onnxruntime)"
            ) from exc
        self._np = np

        self._tokenizer = load_tokenizer(
            base / "tokenizer.json", max_length=max_length
        )
        self._session = ort.InferenceSession(
            self.model_file, providers=["CPUExecutionProvider"]
        )
        # get_inputs() (not the removed `.inputs` property) for onnxruntime
        # >= 1.29 compatibility.
        self._session_inputs = {item.name for item in self._session.get_inputs()}
        self._pad_id = self._tokenizer.token_to_id("[PAD]")

    def predict(self, texts: list[str]) -> list[tuple[float, float]]:
        """Tokenize + run the model; softmax over logits.

        With a known ``[PAD]`` token the whole batch is padded and sent in
        one ``session.run``; without one (unexpected for this model), it
        falls back to single-text calls so varying lengths still work.
        """
        if not texts:
            return []
        np = self._np
        if self._pad_id is not None:
            self._tokenizer.enable_padding(
                pad_id=self._pad_id, pad_token="[PAD]", direction="right"
            )
            encoded = self._tokenizer.encode_batch(list(texts))
            feed = self._build_feed(
                np.asarray([enc.ids for enc in encoded], dtype=np.int64),
                np.asarray(
                    [enc.attention_mask for enc in encoded], dtype=np.int64
                ),
            )
            raw_logits = self._session.run(None, feed)[0].tolist()
        else:
            self._tokenizer.no_padding()
            raw_logits = []
            for text in texts:
                enc = self._tokenizer.encode(text)
                feed = self._build_feed(
                    np.asarray([enc.ids], dtype=np.int64),
                    np.asarray([enc.attention_mask], dtype=np.int64),
                )
                raw_logits.extend(self._session.run(None, feed)[0].tolist())
        return [
            (float(row[0]), float(row[1])) for row in softmax_rows(raw_logits)
        ]

    def _build_feed(self, input_ids, attention_mask) -> dict:
        """Build the ORT feed with only the inputs the graph declares."""
        np = self._np
        feed: dict = {}
        if "input_ids" in self._session_inputs:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._session_inputs:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._session_inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        if "position_ids" in self._session_inputs:
            feed["position_ids"] = np.tile(
                np.arange(input_ids.shape[1]), (input_ids.shape[0], 1)
            )
        return feed
