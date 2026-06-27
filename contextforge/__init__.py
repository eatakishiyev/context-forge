"""ContextForge — a context compiler for LLM agents.

Sits between your app and the model: it *scores*, *compresses*, *reorders*, and
*budgets* everything entering the context window, so the model performs as if the
input were short and clean.

Quickstart
----------
    from contextforge import ContextCompiler, Trace

    trace = Trace.load("examples/sample_trace.json")
    result = ContextCompiler(target_tokens=30_000).compile(trace)
    print(result.summary())

See ``contextforge.cli`` for the command line interface.
"""

from .types import ContextItem, Trace, CompileResult, Action
from .compiler import ContextCompiler, Policy
from .rot import rot_score, RotReport

__all__ = [
    "ContextItem",
    "Trace",
    "CompileResult",
    "Action",
    "ContextCompiler",
    "Policy",
    "rot_score",
    "RotReport",
]

__version__ = "0.1.0"
