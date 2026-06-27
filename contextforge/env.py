"""Minimal .env loader (stdlib only).

Loads KEY=VALUE pairs from .env.local / .env (searched from the cwd up to the
repo root) into os.environ, *without* overriding variables already set in the
real environment. Keeps secrets in a gitignored file instead of the shell rc.
"""

from __future__ import annotations

import os
from typing import Optional

_LOADED = False


def _parse_into_environ(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:  # never override the real env
                os.environ[key] = val


def load_dotenv(start: Optional[str] = None, force: bool = False) -> None:
    """Load .env.local then .env, walking up from ``start`` (default: cwd)."""
    global _LOADED
    if _LOADED and not force:
        return
    _LOADED = True

    seen = set()
    here = os.path.abspath(start or os.getcwd())
    # also include the package's repo root so it works from any cwd
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = [here, repo_root]

    dirs = []
    for root in roots:
        d = root
        while d and d not in dirs:
            dirs.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    for d in dirs:
        for name in (".env.local", ".env"):
            p = os.path.join(d, name)
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                _parse_into_environ(p)
