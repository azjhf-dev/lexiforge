"""Build a tiny synthetic Llama-arch GGUF for tests."""

import numpy as np

from lexiforge.gguf import (T_ARRAY, T_BOOL, T_FLOAT32, T_INT64, T_STRING,
                            T_UINT32, write_gguf)

EMBD, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN = 32, 2, 4, 2, 8, 64
N_BYTE = 256
VOCAB = N_BYTE + 3  # byte tokens + "he" + bos + eos
TOK_HE, BOS, EOS = 256, 257, 258


def build_tokens():
    tokens = [f"<0x{i:02X}>" for i in range(N_BYTE)]
    tokens += ["he", "<s>", "</s>"]
    scores = [float(i) for i in range(N_BYTE)] + [1.0, 0.0, 0.0]
    return tokens, scores


def build_model(path, seed=0):
    rng = np.random.default_rng(seed)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def t(name, arr):  # arr in numpy shape; ggml dims are reversed
        return name, (tuple(arr.shape[::-1]), 0, arr.tobytes())  # 0 = F32

    tensors = {}
    for name, payload in [
        t("token_embd.weight", w(VOCAB, EMBD)),
        t("output_norm.weight", w(EMBD)),
        t("output.weight", w(VOCAB, EMBD)),
    ]:
        tensors[name] = payload
    for i in range(LAYERS):
        b = f"blk.{i}."
        for name, arr in [
            (b + "attn_norm.weight", w(EMBD)),
            (b + "attn_q.weight", w(HEADS * HEAD_DIM, EMBD)),
            (b + "attn_k.weight", w(KV_HEADS * HEAD_DIM, EMBD)),
            (b + "attn_v.weight", w(KV_HEADS * HEAD_DIM, EMBD)),
            (b + "attn_output.weight", w(EMBD, HEADS * HEAD_DIM)),
            (b + "ffn_norm.weight", w(EMBD)),
            (b + "ffn_gate.weight", w(FFN, EMBD)),
            (b + "ffn_up.weight", w(FFN, EMBD)),
            (b + "ffn_down.weight", w(EMBD, FFN)),
        ]:
            tensors[name] = (arr.shape[::-1], 0, arr.tobytes())

    tokens, scores = build_tokens()
    md = {
        "general.architecture": (T_STRING, "llama"),
        "llama.embedding_length": (T_UINT32, EMBD),
        "llama.block_count": (T_UINT32, LAYERS),
        "llama.attention.head_count": (T_UINT32, HEADS),
        "llama.attention.head_count_kv": (T_UINT32, KV_HEADS),
        "llama.feed_forward_length": (T_UINT32, FFN),
        "llama.rope.freq_base": (T_FLOAT32, 10000.0),
        "llama.attention.layer_norm_rms_epsilon": (T_FLOAT32, 1e-5),
        "llama.context_length": (T_UINT32, 128),
        "tokenizer.ggml.model": (T_STRING, "gpt2"),
        "tokenizer.ggml.tokens": (T_ARRAY, (T_STRING, tokens)),
        "tokenizer.ggml.scores": (T_ARRAY, (T_FLOAT32, scores)),
        "tokenizer.ggml.bos_token_id": (T_INT64, BOS),
        "tokenizer.ggml.eos_token_id": (T_INT64, EOS),
        "tokenizer.ggml.add_bos_token": (T_BOOL, False),
    }
    write_gguf(path, md, tensors)
    return path
