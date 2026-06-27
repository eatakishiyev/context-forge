"""Drop-in compiling proxy — zero-code adoption.

Point your SDK's ``base_url`` at this server. It parses each chat request,
compiles the context (score → compress → reorder → budget), forwards the smaller
request upstream, and returns the upstream response unchanged — with the rot/token
deltas attached as ``x-contextforge-*`` response headers.

    contextforge proxy --api anthropic --port 8788 --budget 30000
    # then in your client:
    #   Anthropic(base_url="http://localhost:8788")

Stdlib only (http.server + urllib) so the core stays dependency-free.

Limitations (v0): non-streaming only (stream is downgraded); content blocks are
flattened to text. Use --dry-run to inspect the compiled request without an API
key or any upstream call.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .adapters import ADAPTERS
from .compiler import ContextCompiler, Policy
from .profiles import get_profile

UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

# Hop-by-hop / length headers we must not blindly forward.
_SKIP_HEADERS = {"host", "content-length", "accept-encoding", "connection"}


class ProxyConfig:
    def __init__(self, api: str, upstream: str, budget: Optional[int],
                 dry_run: bool, profiles_path: Optional[str]):
        self.api = api
        self.upstream = upstream.rstrip("/")
        self.budget = budget
        self.dry_run = dry_run
        self.profiles_path = profiles_path
        self.to_trace, self.to_request, self.path = ADAPTERS[api]


def _compile_body(cfg: ProxyConfig, body: dict):
    trace = cfg.to_trace(body)
    policy = Policy(target_tokens=cfg.budget)
    policy.apply_profile(get_profile(trace.model, cfg.profiles_path))
    result = ContextCompiler(policy=policy).compile(trace)
    new_body = cfg.to_request(result, body)
    return new_body, result


def make_handler(cfg: ProxyConfig):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logging
            sys.stderr.write("  [proxy] " + (fmt % args) + "\n")

        def _send_json(self, status, obj, extra_headers=None):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "api": cfg.api,
                                      "upstream": cfg.upstream,
                                      "dry_run": cfg.dry_run})
            else:
                self._send_json(404, {"error": "GET not supported here"})

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except Exception as e:
                self._send_json(400, {"error": f"invalid JSON: {e}"})
                return

            try:
                new_body, result = _compile_body(cfg, body)
            except Exception as e:
                self._send_json(500, {"error": f"compile failed: {e}"})
                return

            cf_headers = {
                "x-contextforge-rot-before": result.rot_before.total,
                "x-contextforge-rot-after": result.rot_after.total,
                "x-contextforge-tokens-before": result.tokens_before,
                "x-contextforge-tokens-after": result.tokens_after,
                "x-contextforge-tokens-saved": result.tokens_saved,
            }

            if cfg.dry_run:
                self._send_json(200, {
                    "dry_run": True,
                    "report": {k: v for k, v in cf_headers.items()},
                    "actions": [a.to_dict() for a in result.actions],
                    "compiled_request": new_body,
                }, extra_headers=cf_headers)
                return

            status, resp_headers, resp_body = self._forward(new_body)
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() in _SKIP_HEADERS or k.lower() == "transfer-encoding":
                    continue
                self.send_header(k, v)
            for k, v in cf_headers.items():
                self.send_header(k, str(v))
            self.send_header("content-length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _forward(self, new_body: dict):
            url = cfg.upstream + cfg.path
            data = json.dumps(new_body).encode("utf-8")
            fwd_headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _SKIP_HEADERS}
            fwd_headers["content-type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=fwd_headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, dict(resp.headers), resp.read()
            except urllib.error.HTTPError as e:
                return e.code, dict(e.headers), e.read()
            except Exception as e:
                return 502, {"content-type": "application/json"}, \
                    json.dumps({"error": f"upstream request failed: {e}"}).encode()

    return Handler


def serve(api: str = "anthropic", port: int = 8788, budget: Optional[int] = None,
          upstream: Optional[str] = None, dry_run: bool = False,
          profiles_path: Optional[str] = None):
    cfg = ProxyConfig(
        api=api,
        upstream=upstream or UPSTREAMS[api],
        budget=budget,
        dry_run=dry_run,
        profiles_path=profiles_path,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg))
    mode = "DRY-RUN (no upstream calls)" if dry_run else f"-> {cfg.upstream}"
    print(f"ContextForge proxy [{api}] on http://127.0.0.1:{port}  {mode}")
    print(f"  endpoint: POST {cfg.path}   budget: {budget or 'none'}")
    print("  point your client's base_url here. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping proxy.")
        server.shutdown()
