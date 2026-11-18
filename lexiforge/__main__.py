"""Command line entry point.

  python -m lexiforge serve model.gguf [--port 8080]
  python -m lexiforge chat  model.gguf "prompt"
  python -m lexiforge fit   model.gguf --vram 24 [--ctx 32768]
  python -m lexiforge tune  model.gguf
  python -m lexiforge bench-launch model.gguf --preset rdna-dense
  python -m lexiforge info  model.gguf
"""

from __future__ import annotations

import argparse
import sys

from . import fit as fit_mod
from .backends import PRESETS, launch
from .gguf import GGUF
from .model import Model


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lexiforge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="OpenAI-compatible server + web panel")
    s.add_argument("model"); s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)

    c = sub.add_parser("chat", help="one-shot generation in the terminal")
    c.add_argument("model"); c.add_argument("prompt")
    c.add_argument("--max-tokens", type=int, default=64)
    c.add_argument("--temperature", type=float, default=0.0)

    f = sub.add_parser("fit", help="VRAM fit planner")
    f.add_argument("model"); f.add_argument("--vram", type=float, default=None)
    f.add_argument("--ctx", type=int, default=32768)

    t = sub.add_parser("tune", help="benchmark and keep the winning config")
    t.add_argument("model")

    b = sub.add_parser("bench-launch", help="print a tuned llama.cpp Vulkan command")
    b.add_argument("model"); b.add_argument("--preset", choices=sorted(PRESETS),
                                            default="rdna-dense")
    b.add_argument("--ctx", type=int, default=32768)

    i = sub.add_parser("info", help="model metadata summary")
    i.add_argument("model")

    args = ap.parse_args(argv)
    if args.cmd == "serve":
        from .server import serve
        serve(args.model, args.host, args.port)
    elif args.cmd == "chat":
        with Model(args.model) as m:
            out = []
            for tok in m.generate(args.prompt, max_tokens=args.max_tokens,
                                  temperature=args.temperature):
                out.append(tok)
                print(m.tok.decode([tok]), end="", flush=True)
            print()
    elif args.cmd == "fit":
        print(fit_mod.fmt(fit_mod.analyze(args.model, args.vram, args.ctx)))
    elif args.cmd == "tune":
        from .tuner import tune
        tune(args.model)
    elif args.cmd == "bench-launch":
        print(" ".join(launch(args.model, args.preset, ctx=args.ctx)))
    elif args.cmd == "info":
        g = GGUF(args.model)
        print(f"arch: {g.architecture}")
        print(f"tensors: {len(g.tensors)}")
        print(f"size: {g.total_bytes() / (1 << 30):.2f} GiB")
        for k in sorted(g.metadata):
            v = g.metadata[k]
            if not isinstance(v, list) and k.startswith(g.architecture + "."):
                print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
