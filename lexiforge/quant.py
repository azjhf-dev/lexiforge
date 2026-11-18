"""Vectorized dequantizers for ggml quantization formats.

Each dequantizer takes the raw uint8 bytes of a tensor plus its numpy shape
and returns a float32 array. The math follows the public ggml block layouts
(Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F16, F32) and is checked against
hand-computed blocks in tests/.
"""

from __future__ import annotations

import numpy as np

from .gguf import TensorInfo, type_nbytes

QK_K = 256


def _f16_to_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32) if x.dtype != np.float16 else x.astype(np.float32)


def _dequant_f32(raw: np.ndarray, shape) -> np.ndarray:
    return raw.view(np.float32).reshape(shape).astype(np.float32)


def _dequant_f16(raw: np.ndarray, shape) -> np.ndarray:
    return raw.view(np.float16).reshape(shape).astype(np.float32)


def _blocks(raw: np.ndarray, block_bytes: int) -> np.ndarray:
    return raw.reshape(-1, block_bytes)


def _dequant_q8_0(raw: np.ndarray, shape) -> np.ndarray:
    b = _blocks(raw, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)[:, None]
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(shape)


def _dequant_q4_0(raw: np.ndarray, shape) -> np.ndarray:
    b = _blocks(raw, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)[:, None]
    qs = b[:, 2:]  # (n, 16)
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    out = np.concatenate([lo, hi], axis=1) * d
    return out.reshape(shape)


def _k_scales_min(sc: np.ndarray):
    """Unpack the 12-byte packed (scale, min) pairs of K-quants.

    sc: (n, 12) uint8 -> scales (n, 8) float, mins (n, 8) float
    """
    scales = np.empty((sc.shape[0], 8), dtype=np.float32)
    mins = np.empty((sc.shape[0], 8), dtype=np.float32)
    lo = sc[:, 0:4]
    mid = sc[:, 4:8]
    hi = sc[:, 8:12]
    scales[:, 0:4] = lo & 63
    mins[:, 0:4] = mid & 63
    scales[:, 4:8] = (hi & 0x0F) | ((lo >> 6) << 4)
    mins[:, 4:8] = (hi >> 4) | ((mid >> 6) << 4)
    return scales, mins


def _dequant_q4_k(raw: np.ndarray, shape) -> np.ndarray:
    b = _blocks(raw, 144)
    n = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    scales, mins = _k_scales_min(b[:, 4:16])
    qs = b[:, 16:].reshape(n, 4, 32)  # 4 pairs of groups, 32 bytes each
    out = np.empty((n, 256), dtype=np.float32)
    # group j (32 values) lives in byte-group j//2: low nibble if j even
    for j in range(8):
        half = qs[:, j // 2]
        vals = (half & 0x0F) if j % 2 == 0 else (half >> 4)
        out[:, j * 32:(j + 1) * 32] = (
            vals.astype(np.float32) * (d * scales[:, j])[:, None]
            - (dmin * mins[:, j])[:, None]
        )
    return out.reshape(shape)


def _dequant_q5_k(raw: np.ndarray, shape) -> np.ndarray:
    b = _blocks(raw, 176)
    n = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    scales, mins = _k_scales_min(b[:, 4:16])
    qh = b[:, 16:48]   # 32 bytes of high bits
    qs = b[:, 48:].reshape(n, 4, 32)
    out = np.empty((n, 256), dtype=np.float32)
    # high bits: qh bit l (l in 0..255) adds 16 to value l
    hbits = np.unpackbits(qh, axis=1, bitorder="little").astype(np.float32) * 16.0
    qs_flat = np.empty((n, 256), dtype=np.float32)
    for j in range(8):
        half = qs[:, j // 2]
        qs_flat[:, j * 32:(j + 1) * 32] = (half & 0x0F) if j % 2 == 0 else (half >> 4)
    q5 = qs_flat + hbits
    for j in range(8):
        out[:, j * 32:(j + 1) * 32] = (
            q5[:, j * 32:(j + 1) * 32] * (d * scales[:, j])[:, None]
            - (dmin * mins[:, j])[:, None]
        )
    return out.reshape(shape)


def _dequant_q6_k(raw: np.ndarray, shape) -> np.ndarray:
    b = _blocks(raw, 210)
    n = b.shape[0]
    out = np.empty((n, 256), dtype=np.float32)
    for half in range(2):
        base = half * 128
        ql = b[:, half * 64:half * 64 + 64]          # 64 bytes lower nibbles
        qh = b[:, 128 + half * 32:128 + half * 32 + 32]  # 32 bytes high 2 bits
        sc = b[:, 192 + half * 8:192 + half * 8 + 8].view(np.int8).astype(np.float32)
        d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
        q1 = ((ql[:, :32] & 0x0F) | ((qh & 0x03) << 4)).astype(np.float32) - 32.0
        q2 = ((ql[:, 32:] & 0x0F) | (((qh >> 2) & 0x03) << 4)).astype(np.float32) - 32.0
        q3 = ((ql[:, :32] >> 4) | (((qh >> 4) & 0x03) << 4)).astype(np.float32) - 32.0
        q4 = ((ql[:, 32:] >> 4) | (((qh >> 6) & 0x03) << 4)).astype(np.float32) - 32.0
        for l in range(32):
            isg = l // 16
            out[:, base + l] = d * sc[:, isg + 0] * q1[:, l]
            out[:, base + 32 + l] = d * sc[:, isg + 2] * q2[:, l]
            out[:, base + 64 + l] = d * sc[:, isg + 4] * q3[:, l]
            out[:, base + 96 + l] = d * sc[:, isg + 6] * q4[:, l]
    return out.reshape(shape)


_DEQUANT = {
    "F32": _dequant_f32,
    "F16": _dequant_f16,
    "Q4_0": _dequant_q4_0,
    "Q8_0": _dequant_q8_0,
    "Q4_K": _dequant_q4_k,
    "Q5_K": _dequant_q5_k,
    "Q6_K": _dequant_q6_k,
}

def dequantize(raw: np.ndarray, info: TensorInfo) -> np.ndarray:
    """Dequantize one tensor's raw bytes to float32 with numpy shape."""
    fn = _DEQUANT.get(info.type_name)
    if fn is None:
        raise ValueError(f"dequantization for {info.type_name} is not implemented")
    return fn(raw, info.shape)


def supported_quants():
    return sorted(_DEQUANT)
