"""Backend tests for the deep path — softmax helpers, stub contract, and a
tiny synthetic ONNX model (built offline via ``onnx.helper``; the real
model binary is never required, network is never touched)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.deep_path import StubBackend, softmax_rows
from src.deep_path.backend import OnnxRuntimeBackend
from src.deep_path.tokenizer_utils import DEFAULT_MAX_LENGTH, load_tokenizer

pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
tokenizers = pytest.importorskip("tokenizers", reason="tokenizers not installed")

VOCAB = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "a": 4,
    "b": 5,
}


# ---------------------------------------------------------------------------
# softmax helper (pure python)
# ---------------------------------------------------------------------------


class TestSoftmaxRows:
    def test_uniform_logits(self) -> None:
        assert softmax_rows([[0.0, 0.0]]) == [[0.5, 0.5]]

    def test_extreme_values_do_not_overflow(self) -> None:
        rows = softmax_rows([[1000.0, -1000.0]])
        assert rows[0][0] == pytest.approx(1.0, abs=1e-9)
        assert rows[0][1] == pytest.approx(0.0, abs=1e-9)

    def test_rows_sum_to_one(self) -> None:
        rows = softmax_rows([[1.0, 2.0], [-5.0, 5.0, 0.5]])
        for row in rows:
            assert sum(row) == pytest.approx(1.0)

    def test_known_values(self) -> None:
        row = softmax_rows([[0.0, math.log(3.0)]])[0]
        assert row[1] == pytest.approx(0.75)

    def test_empty_row(self) -> None:
        assert softmax_rows([[]]) == [[]]


# ---------------------------------------------------------------------------
# StubBackend contract
# ---------------------------------------------------------------------------


class TestStubBackend:
    def test_predict_returns_pair_per_text(self) -> None:
        stub = StubBackend(scores=(0.3, 0.7))
        assert stub.predict(["x", "y", "z"]) == [(0.3, 0.7)] * 3

    def test_predict_empty_list(self) -> None:
        stub = StubBackend()
        assert stub.predict([]) == []
        assert stub.calls == 1

    def test_call_counter_and_seen_texts(self) -> None:
        stub = StubBackend()
        stub.predict(["one"])
        stub.predict(["two", "three"])
        assert stub.calls == 2
        assert stub.seen_texts == [["one"], ["two", "three"]]

    def test_metadata(self) -> None:
        stub = StubBackend()
        assert stub.model_name == "stub"
        assert stub.quantized is False


# ---------------------------------------------------------------------------
# Synthetic ONNX model helpers
# ---------------------------------------------------------------------------


def write_synthetic_tokenizer(path: Path) -> None:
    """WordLevel tokenizer.json with CLS/SEP/PAD specials."""
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(path))


def write_synthetic_model(path: Path, *, with_token_type_ids: bool = False) -> None:
    """ONNX graph: logits = [-n, +n], n = sum(attention_mask) per row.

    Injection probability therefore grows monotonically with the number of
    non-pad tokens — deterministic and checkable without the real model.
    ``token_type_ids`` (when declared) is deliberately unused, exercising
    the "feed only declared inputs" logic.
    """
    onnx = pytest.importorskip("onnx", reason="onnx not installed")
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None]),
    ]
    if with_token_type_ids:
        inputs.append(
            helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, [None, None])
        )
    output = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])]

    neg_one = helper.make_tensor("neg_one", TensorProto.FLOAT, [1], [-1.0])
    nodes = [
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "ReduceSum", ["mask_f"], ["n"], axes=[1], keepdims=0
        ),
        helper.make_node("Unsqueeze", ["n"], ["n_col"], axes=[1]),
        helper.make_node("Mul", ["n_col", "neg_one"], ["neg"]),
        helper.make_node("Concat", ["neg", "n_col"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "synthetic", inputs, output, initializer=[neg_one]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)]
    )
    path.write_bytes(model.SerializeToString())


def make_synthetic_backend(
    tmp_path: Path, *, with_token_type_ids: bool = False
) -> OnnxRuntimeBackend:
    write_synthetic_tokenizer(tmp_path / "tokenizer.json")
    write_synthetic_model(
        tmp_path / "model.onnx", with_token_type_ids=with_token_type_ids
    )
    return OnnxRuntimeBackend(tmp_path)


def expected_pair(non_pad_tokens: int) -> tuple[float, float]:
    pair = math.exp(2 * non_pad_tokens)  # softmax([-n, +n]) ratio
    return (1.0 / (1.0 + pair), pair / (1.0 + pair))


class TestOnnxRuntimeBackendSynthetic:
    def test_metadata_and_model_name(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        assert backend.model_name == tmp_path.name
        assert backend.quantized is False
        assert backend.max_length == DEFAULT_MAX_LENGTH

    def test_predict_single_text(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        # "a" → [CLS] a [SEP] = 3 non-pad tokens → softmax([-3, 3])
        (benign, injection) = backend.predict(["a"])[0]
        assert benign == pytest.approx(expected_pair(3)[0], rel=1e-6)
        assert injection == pytest.approx(expected_pair(3)[1], rel=1e-6)

    def test_predict_batch_pads_different_lengths(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        results = backend.predict(["a b", "a"])
        assert results[0] == pytest.approx(expected_pair(4), rel=1e-6)
        assert results[1] == pytest.approx(expected_pair(3), rel=1e-6)

    def test_predict_preserves_input_order(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        results = backend.predict(["a b", "a b a b a b"])
        assert results[0][1] < results[1][1]

    def test_undeclared_inputs_not_fed(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        assert "token_type_ids" not in backend._session_inputs
        assert backend.predict(["a b c"])  # runs cleanly

    def test_declared_token_type_ids_is_fed(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path, with_token_type_ids=True)
        (benign, injection) = backend.predict(["a"])[0]
        assert injection == pytest.approx(expected_pair(3)[1], rel=1e-6)

    def test_truncation_caps_at_max_length(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        backend.max_length = 16  # tiny cap so the test stays cheap
        backend._tokenizer.enable_truncation(max_length=16)
        long_text = "a " * 100
        (benign, injection) = backend.predict([long_text])[0]
        # 16 tokens incl. CLS/SEP → 16 non-pad tokens
        assert injection == pytest.approx(expected_pair(16)[1], rel=1e-6)

    def test_empty_batch(self, tmp_path: Path) -> None:
        backend = make_synthetic_backend(tmp_path)
        assert backend.predict([]) == []


class TestOnnxRuntimeBackendErrors:
    def test_missing_model_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="download_model"):
            OnnxRuntimeBackend(tmp_path)

    def test_missing_tokenizer(self, tmp_path: Path) -> None:
        write_synthetic_model(tmp_path / "model.onnx")
        with pytest.raises(FileNotFoundError, match="download_model"):
            OnnxRuntimeBackend(tmp_path)


# ---------------------------------------------------------------------------
# Tokenizer helpers (no ONNX needed)
# ---------------------------------------------------------------------------


class TestTokenizerUtils:
    def test_load_tokenizer_truncates_to_max_length(self, tmp_path: Path) -> None:
        write_synthetic_tokenizer(tmp_path / "tokenizer.json")
        tok = load_tokenizer(tmp_path / "tokenizer.json", max_length=32)
        encoding = tok.encode("a " * 500)
        assert len(encoding.ids) <= 32

    def test_load_tokenizer_attaches_cls_sep(self, tmp_path: Path) -> None:
        write_synthetic_tokenizer(tmp_path / "tokenizer.json")
        tok = load_tokenizer(tmp_path / "tokenizer.json", max_length=16)
        ids = tok.encode("a b").ids
        assert ids[0] == VOCAB["[CLS]"]
        assert ids[-1] == VOCAB["[SEP]"]

    def test_load_tokenizer_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="download_model"):
            load_tokenizer(tmp_path / "tokenizer.json")


# ---------------------------------------------------------------------------
# Real (downloaded) model — skipped when models/ is absent; no network.
# ---------------------------------------------------------------------------

REAL_MODEL_DIR = Path("models/deberta-v3-base-prompt-injection-v2")


@pytest.mark.skipif(
    not REAL_MODEL_DIR.is_dir() or not (REAL_MODEL_DIR / "model.onnx").is_file(),
    reason="real model not downloaded (run scripts/download_model.py)",
)
class TestRealBackendSmoke:
    def test_real_model_scores_shape(self) -> None:
        backend = OnnxRuntimeBackend(REAL_MODEL_DIR)
        assert backend.quantized is False  # v2 repo ships no quantized variant
        results = backend.predict(["hello there"])
        (benign, injection) = results[0]
        assert 0.0 <= benign <= 1.0
        assert 0.0 <= injection <= 1.0
        assert benign + injection == pytest.approx(1.0)
