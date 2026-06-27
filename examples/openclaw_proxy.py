"""ContextForge × OpenClaw — keep a 24/7 agent sharp (and cheap) as context grows.

OpenClaw (https://openclaw.ai) is an always-on personal agent: persistent memory,
100+ connected services, autonomous task loops. That's the *exact* profile where
context rot bites — a session that never ends accumulates a huge, tangled context,
and the model quietly starts forgetting the user's standing rules.

This script synthesizes a realistic long OpenClaw session (persona + persistent
memory + a wall of tool output from Gmail/Calendar/GitHub syncs, with one
load-bearing standing rule buried in the middle), then runs the ContextForge
compiler on it and reports the rot + token deltas — and checks the buried rule
survived into what the model actually sees.

Run:  python examples/openclaw_proxy.py

In production you don't change OpenClaw's code at all — you point its model
base_url at the ContextForge proxy (see the bottom of this file).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextforge import ContextCompiler, Trace, ContextItem  # noqa: E402

# The standing rule the agent MUST still honor at turn 200 — buried mid-session.
STANDING_RULE = (
    "Standing rule from the user: NEVER schedule meetings or appointments on "
    "Fridays — Friday is reserved deep-work time. Always offer Mon–Thu slots."
)

SERVICES = ["gmail", "gcal", "github", "slack", "spotify", "obsidian",
            "browser", "whatsapp", "drive", "twitter"]


def _sync_dump(seed: int, lines: int = 90) -> str:
    rows = []
    for d in range(lines):
        k = seed * 97 + d * 31
        svc = SERVICES[k % len(SERVICES)]
        rows.append(
            f"[{svc}] event#{100000+k} ts=2026-06-{10+(k%18):02d}T{k%24:02d}:{k%60:02d} "
            f"action=sync status=ok items={1+k%37} cursor=cur_{k%9973} "
            f"note=routine background poll, nothing requiring user attention"
        )
    return "\n".join(rows)


def build_openclaw_session(filler_blocks: int = 28) -> Trace:
    items = [
        ContextItem(
            role="system",
            content="You are the user's OpenClaw assistant. You act across their "
                    "connected services, remember their preferences, and run tasks "
                    "autonomously. Always respect the user's standing rules.",
            pinned=True,
        ),
        # Persistent memory OpenClaw carries across the 24/7 session.
        ContextItem(role="assistant", kind="memory",
                    content="User's home airport is SFO; prefers aisle seats."),
        ContextItem(role="assistant", kind="memory",
                    content="User's working hours are 9am–6pm Pacific."),
    ]

    # First half: a wall of low-salience background sync output.
    for i in range(filler_blocks):
        items.append(ContextItem(role="tool", kind="tool_result",
                                 content=f"BACKGROUND SYNC #{i}\n" + _sync_dump(i)))

    # >>> the load-bearing standing rule, buried in the middle of the session
    items.append(ContextItem(role="user", kind="fact", content=STANDING_RULE))

    # Second half: more background noise pushes the rule into the dead center.
    for i in range(filler_blocks):
        items.append(ContextItem(role="tool", kind="tool_result",
                                 content=f"BACKGROUND SYNC #{100+i}\n" + _sync_dump(100 + i)))
        if i % 5 == 0:
            items.append(ContextItem(role="user",
                         content="btw what's a good playlist for focus?"))
            items.append(ContextItem(role="assistant",
                         content="Queued your 'Deep Focus' playlist on Spotify."))

    # The live task at the very end.
    items.append(ContextItem(
        role="user",
        content="Book me a dentist cleaning next week and add it to my calendar."))

    return Trace(
        items=items,
        task="Book a dentist appointment next week and add it to the calendar.",
        model="claude-opus-4-8",
    )


def main() -> int:
    trace = build_openclaw_session()
    # Budget a generous-but-bounded window; compress everything else away.
    result = ContextCompiler(target_tokens=20_000).compile(trace)

    print("ContextForge × OpenClaw — long-session compile")
    print("=" * 56)
    print(result.summary())

    seen = "\n".join(i.content for i in result.items)
    kept = "Fridays" in seen
    print()
    print(f"buried standing rule ('no Fridays') still visible to the model: "
          f"{'YES ✓' if kept else 'NO ✗'}")
    anchored = any(i.meta.get("cf_generated") and "Friday" in i.content
                   for i in result.items)
    print(f"…and lifted into an edge anchor (not stranded mid-context): "
          f"{'YES ✓' if anchored else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Production wiring — zero code changes inside OpenClaw
# ---------------------------------------------------------------------------
# 1. Run the compiling proxy next to OpenClaw:
#
#       contextforge proxy --api anthropic --budget 20000     # or --api openai
#
# 2. Point OpenClaw's model provider base_url at it, e.g. in your OpenClaw
#    model config / env:
#
#       ANTHROPIC_BASE_URL=http://localhost:8788
#       # or for an OpenAI-backed OpenClaw model:  OPENAI_BASE_URL=http://localhost:8788
#
# Every turn OpenClaw sends is now compiled before it hits the model, and each
# response carries x-contextforge-rot-before/after and x-contextforge-tokens-*
# headers so you can watch rot and spend per call.
