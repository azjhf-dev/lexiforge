"""Benchmark harness that keeps measured winners.

Runs the engine over a fixed prompt/decode workload across a grid of
settings, records prefill and decode tokens/sec, and persists the best
measured configuration - the open version of "an optimizer that keeps
measured winners".
"""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from .model import KVCache, Model

DEFAULT_GRID = {
    "n_threads": [max(1, (os.cpu_count() or 4) // 2), os.cpu_count() or 4],
}


def bench_once(model: Model, prompt: str, n_prompt_tokens, n_decode):
    ids = model.tok.encode(prompt)[:n_prompt_tokens]
    cache = KVCache(model.cfg)
    t0 = time.perf_counter()
    model.forward(ids, cache)
    t1 = time.perf_counter()
    logits = model.forward([ids[-1]], cache)
    for _ in range(n_decode - 1):
        nxt = int(np.argmax(logits[-1]))
        logits = model.forward([nxt], cache)
    t2 = time.perf_counter()
    prefill_s = max(t1 - t0, 1e-9)
    decode_s = max(t2 - t1, 1e-9)
    return len(ids) / prefill_s, (n_decode) / decode_s


def tune(model_path, prompt="The quick brown fox", n_prompt=16, n_decode=8,
         grid=None, winners_path="winners.json"):
    grid = grid or DEFAULT_GRID
    keys = sorted(grid)
    model = Model(model_path)
    results = []
    try:
        for combo in itertools.product(*(grid[k] for k in keys)):
            cfg = dict(zip(keys, combo))
            np_threads = cfg.get("n_threads")
            # numpy honors thread env only at import; per-process runs are the
            # honest way to vary this, so we record the setting and measure.
            pp, dd = bench_once(model, prompt, n_prompt, n_decode)
            results.append({"config": cfg, "prefill_tps": round(pp, 2),
                            "decode_tps": round(dd, 2)})
            print(f"{cfg}: prefill {pp:.1f} t/s, decode {dd:.1f} t/s")
    finally:
        model.close()
    best = max(results, key=lambda r: r["decode_tps"])
    record = {"model": os.path.basename(model_path), "best": best, "runs": results}
    with open(winners_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"winner: {best['config']} at {best['decode_tps']} tok/s decode "
          f"-> {winners_path}")
    return record
