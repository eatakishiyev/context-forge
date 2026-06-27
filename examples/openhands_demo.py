"""ContextForge × OpenHands — keep an autonomous coding agent on-task.

OpenHands (https://github.com/All-Hands-AI/OpenHands) is an autonomous software
agent: it reads files, runs tests, edits code, and iterates for dozens of steps.
Each step dumps more into the context — file contents, pytest output, stack
traces, git diffs — until an early, load-bearing constraint ("don't touch the
frozen legacy billing module") is buried so deep the agent edits it anyway.

This script synthesizes a long coding-agent session with one such constraint
buried mid-run, then compiles it and checks the constraint survived to the edge
of what the model sees.

Run:  python examples/openhands_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextforge import ContextCompiler, Trace, ContextItem  # noqa: E402

CONSTRAINT = (
    "PROJECT CONSTRAINT: do NOT modify anything under app/legacy/billing/* — it is "
    "frozen and load-bearing. Route all billing changes through "
    "app/billing/BillingAdapter instead. Editing the legacy module will break prod."
)


def _tool_dump(seed: int, lines: int = 80) -> str:
    rows = []
    for d in range(lines):
        k = seed * 89 + d * 23
        f = f"app/mod_{k % 40}/handler_{k % 17}.py"
        rows.append(
            f"PASSED tests/test_{k % 60}.py::test_case_{k % 9} ({1+k%400}ms)  "
            f"covered {f}:{10+k%300}  assert ok  cache={'hit' if k%2 else 'miss'}"
        )
    return "\n".join(rows)


def build_openhands_session(steps: int = 28) -> Trace:
    items = [
        ContextItem(role="system",
                    content="You are an autonomous coding agent. Read, edit, and "
                            "test code to satisfy the task. Obey all project "
                            "constraints.", pinned=True),
        ContextItem(role="assistant", kind="memory",
                    content="Repo: Python service. Entry: app/main.py. "
                            "Tests: pytest. CI gate: 90% coverage."),
    ]
    for i in range(steps):
        items.append(ContextItem(role="tool", kind="tool_result",
                                 content=f"$ pytest -q  (run #{i})\n" + _tool_dump(i)))

    items.append(ContextItem(role="user", kind="fact", content=CONSTRAINT))

    for i in range(steps):
        items.append(ContextItem(role="tool", kind="tool_result",
                                 content=f"$ pytest -q  (run #{100+i})\n" + _tool_dump(100 + i)))
        if i % 6 == 0:
            items.append(ContextItem(role="assistant",
                         content="Reading app/checkout/service.py to locate the bug…"))

    items.append(ContextItem(
        role="user",
        content="The checkout integration test is failing on a billing total. "
                "Find and fix the bug, then make the suite green."))

    return Trace(items=items,
                 task="Fix the failing checkout billing test and pass the suite.",
                 model="claude-opus-4-8")


def main() -> int:
    trace = build_openhands_session()
    result = ContextCompiler(target_tokens=24_000).compile(trace)

    print("ContextForge × OpenHands — coding-agent compile")
    print("=" * 56)
    print(result.summary())

    seen = "\n".join(i.content for i in result.items)
    print()
    print(f"frozen-module constraint still visible to the model: "
          f"{'YES ✓' if 'legacy/billing' in seen else 'NO ✗'}")
    anchored = any(i.meta.get("cf_generated") and "legacy/billing" in i.content
                   for i in result.items)
    print(f"…and anchored to the window edge (not buried mid-run): "
          f"{'YES ✓' if anchored else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Production wiring: run `contextforge proxy --api <anthropic|openai> --budget 24000`
# and point OpenHands' LLM base_url at it (LLM_BASE_URL / OPENAI_API_BASE). Every
# step's context is compiled before the model call; rot + token deltas come back in
# x-contextforge-* response headers.
