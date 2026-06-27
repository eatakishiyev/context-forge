"""ContextForge web UI server — "what the model actually saw".

A dependency-free (stdlib only) HTTP server that exposes the compiler as JSON
endpoints and serves a single-page dashboard:

    GET  /                serve the SPA
    GET  /api/sample      the bundled sample trace
    POST /api/score       {trace, model} -> rot report
    POST /api/compile     {trace, budget, model, passes} -> full compile result

    contextforge ui --port 8799 [--open]
"""

from __future__ import annotations

import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .compiler import ContextCompiler, Policy
from .profiles import get_profile
from .rot import rot_score
from .salience import score_salience
from .types import Trace

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _index_html() -> bytes:
    with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
        return f.read()


def _sample_trace() -> dict:
    path = os.path.join(REPO_ROOT, "examples", "sample_trace.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
_CTYPES = {".png": "image/png", ".ico": "image/x-icon", ".svg": "image/svg+xml",
           ".webmanifest": "application/manifest+json", ".json": "application/json"}


def _static(req_path: str):
    """Serve a file from assets/ only — no path traversal."""
    rel = os.path.normpath(req_path.lstrip("/"))
    path = os.path.join(REPO_ROOT, rel)
    try:
        inside = os.path.commonpath([os.path.abspath(path), ASSETS_DIR]) == ASSETS_DIR
    except ValueError:
        return None
    if not (inside and os.path.isfile(path)):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return data, _CTYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


def _score(payload: dict) -> dict:
    trace = Trace.from_dict(payload.get("trace", {}))
    model = payload.get("model") or trace.model
    profile = get_profile(model, payload.get("profiles"))
    score_salience(trace.items, trace.task)
    report = rot_score(trace.items, trace.task, model, **profile.rot_kwargs())
    return {
        "report": report.to_dict(),
        "profile": {"name": profile.name, "danger_start": profile.danger_start,
                    "danger_full": profile.danger_full},
    }


def _compile(payload: dict) -> dict:
    trace = Trace.from_dict(payload.get("trace", {}))
    passes = payload.get("passes", {}) or {}
    policy = Policy(
        target_tokens=payload.get("budget") or None,
        dedup=passes.get("dedup", True),
        truncate=passes.get("truncate", True),
        reorder=passes.get("reorder", True),
        anchor=passes.get("anchor", True),
    )
    model = payload.get("model") or trace.model
    policy.apply_profile(get_profile(model, payload.get("profiles")))
    result = ContextCompiler(policy=policy).compile(trace)
    out = result.to_dict()
    out["n_items_before"] = len(trace.items)
    out["n_items_after"] = len(result.items)
    out["profile"] = policy.danger_start, policy.danger_full
    return out


ROUTES = {"/api/score": _score, "/api/compile": _compile}


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, status, body: bytes, ctype="application/json"):
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status, obj):
            self._send(status, json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _index_html(), "text/html; charset=utf-8")
            elif self.path == "/api/sample":
                self._send_json(200, _sample_trace())
            elif self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                req = self.path.split("?", 1)[0]
                aliases = {"/favicon.ico": "/assets/favicon.ico",
                           "/manifest.webmanifest": "/assets/manifest.webmanifest"}
                target = aliases.get(req, req if req.startswith("/assets/") else None)
                served = _static(target) if target else None
                if served:
                    self._send(200, served[0], served[1])
                else:
                    self._send_json(404, {"error": "not found"})

        def do_POST(self):
            fn = ROUTES.get(self.path)
            if fn is None:
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except Exception as e:
                self._send_json(400, {"error": f"invalid JSON: {e}"})
                return
            try:
                self._send_json(200, fn(payload))
            except Exception as e:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(port: int = 8799, open_browser: bool = False):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler())
    url = f"http://127.0.0.1:{port}"
    print(f"ContextForge UI  ->  {url}")
    print("  what the model actually saw. Ctrl-C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping UI.")
        server.shutdown()
