"""Byte-level BPE tokenizer built from GGUF tokenizer metadata.

llama.cpp-family models store the vocab as tokenizer.ggml.tokens plus a
parallel score array. Tokens may be literal text ("he", " world") or byte
escapes ("<0x68>"). Merging works on decoded byte content: start from the
bytes of the input, repeatedly merge the adjacent pair whose concatenation
exists in the vocab with the lowest score (highest BPE priority).
"""

from __future__ import annotations

import re

_BYTE_RE = re.compile(rb"^<0x([0-9A-Fa-f]{2})>$")


class Tokenizer:
    def __init__(self, metadata: dict):
        gm = "tokenizer.ggml."
        self.tokens = metadata[gm + "tokens"]
        self.scores = metadata.get(gm + "scores", [float(i) for i in range(len(self.tokens))])
        self.bos_id = int(metadata.get(gm + "bos_token_id", -1))
        self.eos_id = int(metadata.get(gm + "eos_token_id", -1))
        self.pad_id = int(metadata.get(gm + "padding_token_id", -1))
        self.add_bos = bool(metadata.get(gm + "add_bos_token", True))
        self.model = metadata.get(gm + "model", "gpt2")
        self._bytes = []          # token id -> raw bytes
        self._id_of = {}          # raw bytes -> token id (first wins)
        self._rank = {}           # raw bytes -> merge priority (lower = earlier)
        for i, t in enumerate(self.tokens):
            m = _BYTE_RE.match(t.encode())
            b = bytes([int(m.group(1), 16)]) if m else t.encode("utf-8")
            self._bytes.append(b)
            if b not in self._id_of:
                self._id_of[b] = i
                self._rank[b] = float(self.scores[i])

    # ---- encoding ----

    def _split_text(self, text: str):
        # gpt2-style pre-tokenization keeps merges inside word pieces
        return re.findall(r" ?\w+| ?\d+| ?[^\s\w]+|\s+", text) or [text]

    def _encode_piece(self, piece: str):
        ids = []
        for b in piece.encode("utf-8"):
            tok = self._id_of.get(bytes([b]))
            if tok is None:
                raise ValueError(f"byte 0x{b:02x} has no token in vocab")
            ids.append(tok)
        while len(ids) > 1:
            best, best_rank = -1, float("inf")
            for i in range(len(ids) - 1):
                cand = self._bytes[ids[i]] + self._bytes[ids[i + 1]]
                r = self._rank.get(cand)
                if r is not None and r < best_rank and len(cand) > max(
                        len(self._bytes[ids[i]]), len(self._bytes[ids[i + 1]])):
                    best, best_rank = i, r
            if best < 0:
                break
            ids[best:best + 2] = [self._id_of[
                self._bytes[ids[best]] + self._bytes[ids[best + 1]]]]
        return ids

    def encode(self, text: str, add_bos: bool | None = None):
        ids = []
        for piece in self._split_text(text):
            ids.extend(self._encode_piece(piece))
        if add_bos is None:
            add_bos = self.add_bos
        if add_bos and self.bos_id >= 0:
            ids.insert(0, self.bos_id)
        return ids

    # ---- decoding ----

    def decode(self, ids, skip_special=True) -> str:
        specials = {self.bos_id, self.eos_id, self.pad_id}
        out = bytearray()
        for i in ids:
            if skip_special and i in specials:
                continue
            out += self._bytes[i]
        return out.decode("utf-8", errors="replace")
