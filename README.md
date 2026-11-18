# LexiForge

An open, hackable inference stack for AMD cards and Qwen models.

Inspired by [LexiSuite](https://huggingface.co/Wa1k3r/LexiSuite) - a closed-source,
boot-from-USB AI appliance (OS, kernel, engine, control panel, ~2M lines of Rust)
whose author claims it beats llama.cpp and Strata on RDNA cards. LexiSuite is not
open source, so this project is a clean-room, MIT-licensed take on its open core:
**an inference engine you can actually read, plus the tooling around it.**

LexiForge is not a 2-million-line appliance. It is the part worth having open,
in ~2,000 lines of dependency-light Python:

| Piece | What it does |
| --- | --- |
| `lexiforge.gguf` | Pure-Python GGUF v3 container reader/writer (from the public spec) |
| `lexiforge.quant` | Vectorized dequantizers: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F16/F32 |
| `lexiforge.tokenizer` | Byte-level BPE from GGUF tokenizer metadata |
| `lexiforge.model` | From-scratch numpy transformer engine: RMSNorm, RoPE, GQA, SwiGLU, Qwen3 QK-norm, sparse MoE with top-k router |
| `lexiforge.server` | OpenAI-compatible HTTP API (`/v1/chat/completions` with SSE streaming) + bundled web chat panel |
| `lexiforge.fit` | VRAM fit planner: which quant fits your card, KV-cache cost, MoE expert-offload math |
| `lexiforge.tuner` | Benchmark harness that measures and keeps the winning config |
| `lexiforge.backends` | Tuned llama.cpp Vulkan launch presets for RDNA cards |

## Honest scope

- The numpy engine is a **reference engine**: correct, readable, CPU-speed. It
  exists so the whole stack is open and verifiable end to end. Run small models
  on it; don't expect to serve a 27B from it.
- For real speed on AMD hardware, `lexiforge.backends` drives **llama.cpp's
  Vulkan backend** with tuned presets. That part is an accelerator launcher and
  says so - no pretending otherwise.
- No OS, no kernel, no VM monitor, no 1,162-feature control panel. If you want
  the appliance, LexiSuite exists; if you want open code you can fork, you're in
  the right place.

## Install

```sh
git clone https://github.com/azjhf-dev/lexiforge
cd lexiforge
pip install -e .
```

Only dependency: numpy.

## Quick start

```sh
# inspect a model
python -m lexiforge info Qwen3-4B-Q4_K_M.gguf

# will it fit my card?
python -m lexiforge fit Qwen3.8-Flash-Next-Q4_K_M.gguf --vram 24 --ctx 262144

# generate in the terminal (reference engine, CPU)
python -m lexiforge chat Qwen3-0.6B-Q8_0.gguf "hello" --max-tokens 64

# serve an OpenAI-compatible API + web chat panel
python -m lexiforge serve Qwen3-0.6B-Q8_0.gguf --port 8080
# then open http://localhost:8080 or point any OpenAI client at it

# print the tuned llama.cpp Vulkan command for your card
python -m lexiforge bench-launch model.gguf --preset rdna-dense
python -m lexiforge bench-launch flash-next.gguf --preset rdna-moe-offload
```

### The MoE trick (how one 24 GB card serves a 94 GB MoE)

Models like Qwen3.8-Flash-Next are mostly expert tables: ~51 GB of embeddings
and experts that only a few tokens touch at a time. Keep attention, norms and
the router on the GPU; stream experts from DDR5. `fit` shows the split:

```sh
python -m lexiforge fit flash-next.gguf --vram 24
# weights 94 GiB (12 GiB non-expert, 82 GiB experts)
# fits with expert offload to RAM: True (resident ~14 GiB)
```

and `bench-launch --preset rdna-moe-offload` emits the llama.cpp flags that do it
(`--override-tensor exps=CPU`, flash attention, q8_0 KV cache).

## Tests

```sh
python -m tests.test_engine        # stdlib runner, or: pytest tests/
```

Covers GGUF round-trip, every dequantizer against hand-computed blocks,
tokenizer merges and byte fallback, and KV-cache consistency (incremental
decode must equal joint prefill, the property everything else rests on).

## Layout

```
lexiforge/     the package (engine, server, tools)
panel/         bundled web chat UI
tests/         synthetic-model tests (tiny GGUF built by tests/tiny_model.py)
```

## Credit

- LexiSuite by W61k3r - the closed-source project that prompted this one:
  https://huggingface.co/Wa1k3r/LexiSuite
  (Reddit thread: r/LLMDevs, "Worlds fastest inference engine for AMD cards and
  Qwen models", Nov 2026)
- llama.cpp - the GGUF spec, the block layouts these dequantizers implement,
  and the Vulkan backend the speed presets launch:
  https://github.com/ggml-org/llama.cpp

No LexiSuite code exists in this repository; none is public. Everything here
is original code written against published formats.

## License

MIT. LexiSuite is free for personal use / $100 per year commercial; LexiForge
is just free.
