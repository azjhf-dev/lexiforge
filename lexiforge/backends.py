"""llama.cpp Vulkan launch presets tuned for AMD cards.

The reference engine in lexiforge.model is for reading, hacking and
verifying. For real speed on Radeon hardware, llama.cpp's Vulkan backend is
the fastest open path today; these presets encode the flag combinations that
matter on RDNA cards so you stop hand-editing command lines:

  - flash attention on, quantized KV cache (big context wins)
  - full GPU offload for dense models that fit
  - MoE expert offload to host RAM for models like Qwen3-MoE / Flash-Next
    class, where 50+ GB of expert tables stream fine over DDR5 while the
    card runs attention

Requires a llama.cpp build with -DGGML_VULKAN=ON.
"""

from __future__ import annotations

import shutil
import subprocess

PRESETS = {
    "rdna-dense": [
        "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--n-gpu-layers", "99",
        "--threads", "-1",
    ],
    "rdna-moe-offload": [
        "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--n-gpu-layers", "99",
        "--override-tensor", "exps=CPU",   # experts stay in host RAM
        "--threads", "-1",
    ],
    "cpu-reference": [
        "--n-gpu-layers", "0",
    ],
}


def find_llama_server():
    return shutil.which("llama-server") or shutil.which("llama.cpp-server")


def launch(model_path, preset="rdna-dense", port=8080, ctx=32768, extra=None,
           dry_run=True):
    binpath = find_llama_server()
    args = PRESETS[preset]
    cmd = ([binpath or "llama-server", "--model", model_path,
            "--port", str(port), "--ctx-size", str(ctx)] + list(args)
           + list(extra or []))
    if dry_run or binpath is None:
        return cmd
    return subprocess.Popen(cmd)
