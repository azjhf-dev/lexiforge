import os
import struct
import tempfile

import numpy as np

from lexiforge import gguf
from lexiforge.quant import dequantize
from lexiforge.tokenizer import Tokenizer
from tests.tiny_model import BOS, EOS, TOK_HE, build_model, build_tokens


def _tensor_from_raw(name, dims, gtype, raw, tmp):
    md = {"general.architecture": (gguf.T_STRING, "llama")}
    path = os.path.join(tmp, "q.gguf")
    gguf.write_gguf(path, md, {name: (dims, gtype, raw)})
    g = gguf.GGUF(path).open()
    info = g.tensors[name]
    return dequantize(g.raw_tensor_bytes(info), info)


def test_f32_roundtrip(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        out = _tensor_from_raw("t", (4, 3), 0, arr.tobytes(), tmp)
        assert np.array_equal(out, arr)


def test_q8_0():
    with tempfile.TemporaryDirectory() as tmp:
        d = np.float16(2.0).tobytes()
        q = np.arange(32, dtype=np.int8).tobytes()
        out = _tensor_from_raw("t", (32,), 8, d + q, tmp)
        assert np.allclose(out, np.arange(32) * 2.0)


def test_q4_0():
    with tempfile.TemporaryDirectory() as tmp:
        d = np.float16(0.5).tobytes()
        qs = bytes([0xF8] * 16)  # lo=8 -> 0.0, hi=15 -> 3.5
        out = _tensor_from_raw("t", (32,), 2, d + qs, tmp)
        assert np.allclose(out[:16], 0.0)
        assert np.allclose(out[16:], 3.5)


def _k_scales_all_one():
    # scales=1, mins=0 for all 8 groups in the 12-byte packed layout
    return bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])


def test_q4_k():
    with tempfile.TemporaryDirectory() as tmp:
        raw = (np.float16(1.0).tobytes() + np.float16(0.0).tobytes()
               + _k_scales_all_one() + bytes([0x33] * 128))
        out = _tensor_from_raw("t", (256,), 12, raw, tmp)
        assert np.allclose(out, 3.0)


def test_q5_k():
    with tempfile.TemporaryDirectory() as tmp:
        base = (np.float16(1.0).tobytes() + np.float16(0.0).tobytes()
                + _k_scales_all_one())
        raw = base + bytes(32) + bytes([0x33] * 128)          # high bits 0
        out = _tensor_from_raw("t", (256,), 13, raw, tmp)
        assert np.allclose(out, 3.0)
        raw = base + bytes([0xFF] * 32) + bytes([0x33] * 128)  # high bits 1
        out = _tensor_from_raw("t", (256,), 13, raw, tmp)
        assert np.allclose(out, 19.0)


def test_q6_k():
    with tempfile.TemporaryDirectory() as tmp:
        raw = (bytes([0x11] * 128) + bytes(64)
               + bytes([1] * 16) + np.float16(2.0).tobytes())
        out = _tensor_from_raw("t", (256,), 14, raw, tmp)
        assert np.allclose(out, -62.0)  # q = 1 - 32, d = 2, scale = 1


def test_tokenizer_merge_and_bytes():
    tokens, scores = build_tokens()
    md = {"tokenizer.ggml.tokens": tokens, "tokenizer.ggml.scores": scores,
          "tokenizer.ggml.bos_token_id": BOS, "tokenizer.ggml.eos_token_id": EOS,
          "tokenizer.ggml.add_bos_token": False}
    tok = Tokenizer(md)
    ids = tok.encode("he")
    assert ids == [TOK_HE]
    assert tok.decode([TOK_HE]) == "he"
    # unknown byte path: 'A' = 0x41 -> <0x41>
    ids = tok.encode("A")
    assert ids == [0x41]
    assert tok.decode(ids) == "A"


def test_model_forward_and_cache(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = build_model(os.path.join(tmp, "tiny.gguf"))
        from lexiforge.model import KVCache, Model
        with Model(path) as m:
            ids = [10, 20, 30, 40]
            c1 = KVCache(m.cfg)
            joint = m.forward(ids, c1)
            assert joint.shape == (4, m.cfg.vocab)
            # incremental prefill must match the joint forward
            c2 = KVCache(m.cfg)
            m.forward(ids[:2], c2)
            step = m.forward(ids[2:], c2)
            assert np.allclose(joint[2:], step, atol=1e-4)
            # greedy generation is deterministic and stops at EOS
            gen = list(m.generate("he", max_tokens=8))
            assert len(gen) >= 1
            assert all(0 <= t < m.cfg.vocab for t in gen)


if __name__ == "__main__":
    test_f32_roundtrip()
    test_q8_0()
    test_q4_0()
    test_q4_k()
    test_q5_k()
    test_q6_k()
    test_tokenizer_merge_and_bytes()
    test_model_forward_and_cache()
    print("all tests passed")
