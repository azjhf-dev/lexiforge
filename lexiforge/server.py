"""OpenAI-compatible HTTP server around the reference engine.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (stream and non-stream, SSE)
  POST /v1/completions
  GET  /healthz
  GET  /                      serves the bundled chat panel

Stdlib only: ThreadingHTTPServer + the engine in-process.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Model, apply_chat_template

PANEL = os.path.join(os.path.dirname(__file__), "..", "panel", "index.html")


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "lexiforge/0.1"
    model: Model = None          # set by serve()
    model_id: str = "lexiforge"

    def log_message(self, *a):  # quiet by default
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/healthz":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": self.model_id, "object": "model", "created": int(time.time()),
                 "owned_by": "local"}]})
        if self.path in ("/", "/index.html") and os.path.exists(PANEL):
            with open(PANEL, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = self._body()
        except Exception:
            return self._json(400, {"error": "bad json"})
        if self.path == "/v1/chat/completions":
            return self._chat(req)
        if self.path == "/v1/completions":
            return self._completion(req)
        self._json(404, {"error": "not found"})

    def _gen_args(self, req):
        return dict(
            max_tokens=int(req.get("max_tokens", req.get("max_completion_tokens", 128))),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 1.0)),
            top_k=int(req.get("top_k", 0) or 0),
        )

    def _chat(self, req):
        messages = req.get("messages") or []
        prompt = apply_chat_template(messages)
        ga = self._gen_args(req)
        cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for tok in self.model.generate(prompt, **ga):
                chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": self.model_id,
                         "choices": [{"index": 0, "delta": {
                             "content": self.model.tok.decode([tok])},
                             "finish_reason": None}]}
                self.wfile.write(_sse(chunk))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        ids = list(self.model.generate(prompt, **ga))
        text = self.model.tok.decode(ids)
        self._json(200, {
            "id": cmpl_id, "object": "chat.completion", "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": len(ids),
                      "total_tokens": -1}})

    def _completion(self, req):
        prompt = req.get("prompt", "")
        ga = self._gen_args(req)
        ids = list(self.model.generate(prompt, **ga))
        self._json(200, {
            "id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion",
            "created": int(time.time()), "model": self.model_id,
            "choices": [{"index": 0, "text": self.model.tok.decode(ids),
                         "finish_reason": "stop"}]})


def serve(model_path, host="0.0.0.0", port=8080, model_id=None):
    model = Model(model_path)
    Handler.model = model
    Handler.model_id = model_id or os.path.basename(model_path)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"lexiforge serving {Handler.model_id} on http://{host}:{port}")
    print(f"arch={model.cfg.arch} layers={model.cfg.n_layers} embd={model.cfg.embd} "
          f"moe={model.cfg.is_moe}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        model.close()
