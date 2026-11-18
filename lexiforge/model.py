"""From-scratch numpy transformer engine for GGUF models.

Supports the Llama/Qwen2/Qwen3 family layout: RMSNorm, RoPE, grouped-query
attention, SwiGLU MLP, optional per-head QK norm (Qwen3), and optional
sparse MoE FFNs (Qwen3-MoE style router). Weights are dequantized from the
GGUF file on first use and cached.

This is a reference engine: correct and readable, not fast. For production
speed on AMD hardware use the Vulkan presets in lexiforge.backends, which
drive llama.cpp. The point of this engine is an open core you can read,
modify and measure - no closed blobs.
"""

from __future__ import annotations

import time

import numpy as np

from .gguf import GGUF
from .quant import dequantize
from .tokenizer import Tokenizer


class ModelConfig:
    def __init__(self, md: dict):
        arch = md.get("general.architecture", "llama")
        self.arch = arch
        p = arch + "."
        self.embd = int(md[p + "embedding_length"])
        self.n_layers = int(md[p + "block_count"])
        self.n_heads = int(md[p + "attention.head_count"])
        self.n_kv = int(md.get(p + "attention.head_count_kv", self.n_heads))
        self.ffn = int(md[p + "feed_forward_length"])
        self.rope_theta = float(md.get(p + "rope.freq_base", 10000.0))
        self.rms_eps = float(md.get(p + "attention.layer_norm_rms_epsilon", 1e-6))
        self.ctx = int(md.get(p + "context_length", 4096))
        self.head_dim = int(md.get(p + "attention.key_length", self.embd // self.n_heads))
        self.n_experts = int(md.get(p + "expert_count", 0))
        self.n_experts_used = int(md.get(p + "expert_used_count", 0))
        self.vocab = len(md["tokenizer.ggml.tokens"])

    @property
    def is_moe(self):
        return self.n_experts > 0


def rms_norm(x, w, eps):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def rope(x, pos, theta):
    # x: (..., n_heads, head_dim); apply rotary pairs
    d = x.shape[-1]
    freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    ang = pos[:, None] * freqs[None, :]          # (seq, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    out = np.empty_like(x)
    out[..., :d // 2] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    out[..., d // 2:] = x2 * cos[:, None, :] + x1 * sin[:, None, :]
    return out


def sample(logits, temperature=0.0, top_k=0, top_p=1.0, rng=None):
    logits = logits.astype(np.float64)
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    logits /= temperature
    if top_k > 0 and top_k < logits.size:
        thresh = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= thresh, logits, -np.inf)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        keep = cum <= top_p
        keep[0] = True
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(p.size, p=p))


class KVCache:
    def __init__(self, cfg: ModelConfig):
        self.k = [np.zeros((0, cfg.n_kv, cfg.head_dim), np.float32) for _ in range(cfg.n_layers)]
        self.v = [np.zeros((0, cfg.n_kv, cfg.head_dim), np.float32) for _ in range(cfg.n_layers)]

    def append(self, layer, k, v):
        self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
        self.v[layer] = np.concatenate([self.v[layer], v], axis=0)
        return self.k[layer], self.v[layer]

    def __len__(self):
        return self.k[0].shape[0]


class Model:
    def __init__(self, path, verbose=False):
        self.gguf = GGUF(path).open()
        self.cfg = ModelConfig(self.gguf.metadata)
        self.tok = Tokenizer(self.gguf.metadata)
        self._cache = {}
        self.verbose = verbose

    def close(self):
        self.gguf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- weights ----

    def weight(self, name) -> np.ndarray:
        w = self._cache.get(name)
        if w is None:
            info = self.gguf.tensors.get(name)
            if info is None:
                raise KeyError(f"model has no tensor {name}")
            w = dequantize(self.gguf.raw_tensor_bytes(info), info)
            self._cache[name] = w
        return w

    def has(self, name) -> bool:
        return name in self.gguf.tensors

    def maybe(self, name):
        return self.weight(name) if name in self.gguf.tensors else None

    # ---- forward ----

    def _attention(self, x, layer, cache, pos0):
        cfg = self.cfg
        seq = x.shape[0]
        b = f"blk.{layer}."
        q = x @ self.weight(b + "attn_q.weight").T
        k = x @ self.weight(b + "attn_k.weight").T
        v = x @ self.weight(b + "attn_v.weight").T
        bq, bk, bv = (self.maybe(b + "attn_q.bias"), self.maybe(b + "attn_k.bias"),
                      self.maybe(b + "attn_v.bias"))
        if bq is not None:
            q, k, v = q + bq, k + bk, v + bv
        q = q.reshape(seq, cfg.n_heads, cfg.head_dim)
        k = k.reshape(seq, cfg.n_kv, cfg.head_dim)
        v = v.reshape(seq, cfg.n_kv, cfg.head_dim)
        qn, kn = self.maybe(b + "attn_q_norm.weight"), self.maybe(b + "attn_k_norm.weight")
        if qn is not None:  # Qwen3 per-head QK norm, before RoPE
            q = rms_norm(q, qn, cfg.rms_eps)
            k = rms_norm(k, kn, cfg.rms_eps)
        pos = pos0 + np.arange(seq, dtype=np.float32)
        q = rope(q, pos, cfg.rope_theta)
        k = rope(k, pos, cfg.rope_theta)
        k_all, v_all = cache.append(layer, k, v)
        rep = cfg.n_heads // cfg.n_kv
        if rep > 1:
            k_all = np.repeat(k_all, rep, axis=1)
            v_all = np.repeat(v_all, rep, axis=1)
        # (heads, seq_q, seq_kv) attention with causal mask
        scores = np.einsum("qhd,khd->hqk", q, k_all) / np.sqrt(cfg.head_dim)
        t = k_all.shape[0]
        mask = np.triu(np.ones((seq, t), dtype=bool), k=1 + t - seq)
        scores = np.where(mask[None], -np.inf, scores)
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        attn = np.einsum("hqk,khd->qhd", p, v_all).reshape(seq, cfg.n_heads * cfg.head_dim)
        out = attn @ self.weight(b + "attn_output.weight").T
        bo = self.maybe(b + "attn_output.bias")
        return out + bo if bo is not None else out

    def _ffn(self, x, layer):
        b = f"blk.{layer}."
        if not self.cfg.is_moe:
            gate = x @ self.weight(b + "ffn_gate.weight").T
            up = x @ self.weight(b + "ffn_up.weight").T
            return (self._silu(gate) * up) @ self.weight(b + "ffn_down.weight").T
        # sparse MoE: softmax router, top-k experts, renormalized
        cfg = self.cfg
        logits = x @ self.weight(b + "ffn_gate_inp.weight").T       # (seq, n_experts)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        top = np.argpartition(probs, -cfg.n_experts_used, axis=-1)[:, -cfg.n_experts_used:]
        wg = np.take_along_axis(probs, top, axis=-1)
        wg = wg / wg.sum(axis=-1, keepdims=True)
        ge = self.weight(b + "ffn_gate_exps.weight")   # (n_experts, ffn, embd)
        ue = self.weight(b + "ffn_up_exps.weight")
        de = self.weight(b + "ffn_down_exps.weight")   # (n_experts, embd, ffn)
        out = np.zeros_like(x)
        for slot in range(cfg.n_experts_used):
            idx = top[:, slot]                                # (seq,)
            w = wg[:, slot][:, None]
            g = np.einsum("se,sef->sf", x, ge[idx])
            u = np.einsum("se,sef->sf", x, ue[idx])
            down = np.einsum("sf,sfe->se", self._silu(g) * u, de[idx])
            out += w * down
        return out

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    def forward(self, ids, cache: KVCache) -> np.ndarray:
        cfg = self.cfg
        pos0 = len(cache)
        x = self.weight("token_embd.weight")[ids]
        for layer in range(cfg.n_layers):
            b = f"blk.{layer}."
            h = rms_norm(x, self.weight(b + "attn_norm.weight"), cfg.rms_eps)
            x = x + self._attention(h, layer, cache, pos0)
            h = rms_norm(x, self.weight(b + "ffn_norm.weight"), cfg.rms_eps)
            x = x + self._ffn(h, layer)
        x = rms_norm(x, self.weight("output_norm.weight"), cfg.rms_eps)
        out_w = self.weight("output.weight") if self.has("output.weight") \
            else self.weight("token_embd.weight")
        return x @ out_w.T

    # ---- generation ----

    def generate(self, prompt: str, max_tokens=64, temperature=0.0, top_k=0,
                 top_p=1.0, seed=None, stop_ids=None, stream=False):
        rng = np.random.default_rng(seed)
        cache = KVCache(self.cfg)
        ids = self.tok.encode(prompt)
        logits = self.forward(ids, cache)
        stop = set(stop_ids or ())
        if self.tok.eos_id >= 0:
            stop.add(self.tok.eos_id)
        for _ in range(max_tokens):
            nxt = sample(logits[-1], temperature, top_k, top_p, rng)
            if nxt in stop:
                break
            yield nxt
            logits = self.forward([nxt], cache)

    def chat(self, messages, **kw):
        prompt = apply_chat_template(messages)
        ids = []
        for tok in self.generate(prompt, **kw):
            ids.append(tok)
        return self.tok.decode(ids)


def apply_chat_template(messages) -> str:
    """Minimal ChatML template (Qwen default). For full template fidelity use
    the llama.cpp backend, which renders the model's own jinja template."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
