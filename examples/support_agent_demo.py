"""ContextForge × support agent — the day-in-the-life.

A customer-support agent gives correct answers for the first 30 turns, then starts
getting them wrong after a long chat full of order-history lookups and KB articles.
Nobody touched the prompt — the one load-bearing fact (which card a refund must go
to) rotted in the middle of the window.

This runs the compiler on the bundled support trace and confirms the buried fact
survives into what the model sees.

Run:  python examples/support_agent_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from contextforge import ContextCompiler, Trace  # noqa: E402


def main() -> int:
    trace = Trace.load(os.path.join(HERE, "sample_trace.json"))
    result = ContextCompiler(target_tokens=20_000).compile(trace)

    print("ContextForge × support agent — long-chat compile")
    print("=" * 56)
    print(result.summary())

    seen = "\n".join(i.content for i in result.items)
    ok = ("ending 4417" in seen) and ("ACC-99213" in seen)
    print()
    print(f"buried refund fact (card 4417 / account ACC-99213) still visible: "
          f"{'YES ✓' if ok else 'NO ✗'}")
    print("\nTip: `contextforge bench bench/datasets/sample_suite.json --model "
          "claude-opus-4-8` measures the accuracy delta on this exact case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
