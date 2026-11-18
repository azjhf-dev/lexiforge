"""LexiForge - an open, hackable inference stack for AMD cards and Qwen models.

Components:
  gguf       - pure-Python GGUF v3 container reader/writer
  quant      - vectorized dequantizers (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F16/F32)
  tokenizer  - byte-level BPE from GGUF tokenizer metadata
  model      - from-scratch numpy transformer engine (Llama/Qwen dense + MoE)
  server     - OpenAI-compatible HTTP API around the engine
  fit        - VRAM fit planner: which quant and how many layers fit a card
  tuner      - benchmark harness that keeps measured winners
  backends   - tuned llama.cpp Vulkan launch presets for AMD GPUs
"""

__version__ = "0.1.0"
