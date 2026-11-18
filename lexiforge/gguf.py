"""Minimal GGUF v3 reader/writer built from the public GGUF specification.

No dependency on llama.cpp: this module parses the container itself and
exposes tensor storage as numpy views so the engine can dequantize on its
own. A writer is included so tests can build synthetic models.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_scalar_fmt = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<B",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


# ggml tensor types: type id -> (elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
}
TYPE_IDS = {name: tid for tid, (name, _, _) in GGML_TYPES.items()}


def type_nbytes(ggml_type: int, n_elements: int) -> int:
    name, block_elems, block_bytes = GGML_TYPES[ggml_type]
    if n_elements % block_elems != 0:
        raise ValueError(f"{name} tensor with {n_elements} elements is not a whole number of blocks")
    return n_elements // block_elems * block_bytes


@dataclass
class TensorInfo:
    name: str
    dims: tuple  # ggml order (fastest-changing first); numpy shape is dims[::-1]
    ggml_type: int
    offset: int  # relative to tensor data start

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type][0]

    @property
    def shape(self) -> tuple:
        return self.dims[::-1]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return type_nbytes(self.ggml_type, self.n_elements)


def _read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _read_string(f) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    if vtype == T_STRING:
        return _read_string(f)
    if vtype == T_ARRAY:
        etype = _read(f, "<I")
        n = _read(f, "<Q")
        return [_read_value(f, etype) for _ in range(n)]
    if vtype == T_BOOL:
        return bool(_read(f, "<B"))
    return _read(f, _scalar_fmt[vtype])


def _write_string(buf: bytearray, s: str):
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def _write_value(buf: bytearray, vtype, value):
    if vtype == T_STRING:
        _write_string(buf, value)
    elif vtype == T_ARRAY:
        etype, items = value
        buf += struct.pack("<IQ", etype, len(items))
        for it in items:
            _write_value(buf, etype, it)
    elif vtype == T_BOOL:
        buf += struct.pack("<B", 1 if value else 0)
    else:
        buf += struct.pack(_scalar_fmt[vtype], value)


class GGUF:
    """Parsed GGUF file. Tensor bytes are memory-mapped, not copied."""

    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}
        self.version = 0
        self._data_start = 0
        self._mmap = None
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                raise ValueError(f"{path} is not a GGUF file")
            self.version = _read(f, "<I")
            if self.version not in (2, 3):
                raise ValueError(f"unsupported GGUF version {self.version}")
            n_tensors = _read(f, "<Q")
            n_kv = _read(f, "<Q")
            for _ in range(n_kv):
                key = _read_string(f)
                vtype = _read(f, "<I")
                self.metadata[key] = _read_value(f, vtype)
            infos = []
            for _ in range(n_tensors):
                name = _read_string(f)
                n_dims = _read(f, "<I")
                dims = tuple(_read(f, "<Q") for _ in range(n_dims))
                gtype = _read(f, "<I")
                off = _read(f, "<Q")
                infos.append(TensorInfo(name, dims, gtype, off))
            self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
            # tensor data starts on the next aligned offset
            pos = f.tell()
            self._data_start = pos + (-pos) % self.alignment
        for info in infos:
            if info.ggml_type not in GGML_TYPES:
                raise ValueError(f"tensor {info.name}: unknown ggml type {info.ggml_type}")
            self.tensors[info.name] = info

    def open(self):
        if self._mmap is None:
            self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self

    def close(self):
        if self._mmap is not None:
            self._mmap._mmap.close()
            self._mmap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def raw_tensor_bytes(self, info: TensorInfo) -> np.ndarray:
        """1-D uint8 view of the tensor's stored bytes (zero copy)."""
        start = self._data_start + info.offset
        return self._mmap[start:start + info.nbytes]

    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


def write_gguf(path, metadata: dict, tensors: dict, alignment: int = DEFAULT_ALIGNMENT):
    """Write a GGUF file. `tensors` maps name -> (dims_ggml_order, ggml_type, raw_bytes).

    Metadata values must be tagged: (value_type, value). Arrays are
    (T_ARRAY, (element_type, [items])). Provided for tests and tooling.
    """
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, (vtype, value) in metadata.items():
        _write_string(buf, key)
        buf += struct.pack("<I", vtype)
        _write_value(buf, vtype, value)

    # compute aligned offsets
    offset = 0
    infos = []
    for name, (dims, gtype, raw) in tensors.items():
        offset += (-offset) % alignment
        infos.append((name, dims, gtype, offset, raw))
        offset += len(raw)

    for name, dims, gtype, off, raw in infos:
        _write_string(buf, name)
        buf += struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", gtype, off)

    buf += b"\x00" * ((-len(buf)) % alignment)
    with open(path, "wb") as f:
        f.write(buf)
        pos = 0
        for name, dims, gtype, off, raw in infos:
            pad = off - pos
            if pad:
                f.write(b"\x00" * pad)
            f.write(raw)
            pos = off + len(raw)
    return path


def file_sha256(path, chunk=1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total
