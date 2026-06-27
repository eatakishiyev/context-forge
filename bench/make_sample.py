"""Generate the bundled sample trace + benchmark suite.

Reproducible (no randomness): builds a long, tangled support-agent session with
(a) a load-bearing fact buried in the middle, (b) heavy redundant tool output,
and (c) lots of low-salience filler — the canonical context-rot setup. Run:

    python -m bench.make_sample
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The fact the model must recall at the end — deliberately buried mid-context.
CRITICAL = (
    "Confirmed with billing: the refund of $428.50 for account ACC-99213 must be "
    "issued to the Visa card ending 4417, NOT the card on file ending 2200, "
    "because 2200 was reported stolen on March 3rd."
)


# Roughly how many filler blocks to place on each side of the buried fact.
# Tuned so the trace lands well past the context-rot knee (~250k tokens), where
# real long-context degradation kicks in. Override with CF_SAMPLE_BLOCKS.
BLOCKS_PER_SIDE = int(os.environ.get("CF_SAMPLE_BLOCKS", "26"))


# Rotating vocab so filler blocks are genuinely *different* from each other —
# dedup should NOT trivially collapse them; the realistic savings come from
# truncating stale low-salience material and budgeting, not from duplicate logs.
_SERVICES = ["billing", "auth", "search", "inventory", "shipping", "notify",
             "ledger", "webhook", "session", "media"]
_ROUTES = ["/v1/sync", "/v2/orders", "/health", "/v1/users", "/internal/cron",
           "/v3/events", "/v1/refunds", "/metrics", "/v2/cart", "/v1/index"]
_MSGS = ["connection pool resized", "cache warmed", "snapshot persisted",
         "lease renewed", "shard rebalanced", "token rotated", "batch flushed",
         "replica caught up", "index segment merged", "queue drained"]


def _log_dump(seed: int, lines: int = 110) -> str:
    rows = []
    for d in range(lines):
        k = seed * 131 + d * 17
        svc = _SERVICES[k % len(_SERVICES)]
        route = _ROUTES[(k // 7) % len(_ROUTES)]
        msg = _MSGS[(k // 13) % len(_MSGS)]
        rows.append(
            f"{svc} 2026-06-{10+(k % 18):02d} {k % 24:02d}:{k % 60:02d}:{(k*3) % 60:02d} "
            f"INFO pid=4{k % 9000:04d} req=req_{100000 + k} route={route} "
            f"latency={20 + k % 380}ms status=200 {msg} bytes={1024 + (k % 9000)}"
        )
    return "\n".join(rows)


def _filler_block(i: int) -> dict:
    return {"role": "tool", "kind": "tool_result",
            "content": f"DEBUG TRACE #{i}\n" + _log_dump(i)}


def build_trace(blocks_per_side: int = BLOCKS_PER_SIDE) -> dict:
    items = []
    items.append({
        "role": "system",
        "content": "You are Acme's support agent. Be precise about billing. "
                   "Never issue a refund to a card the customer flagged.",
        "pinned": True,
    })

    # Early small talk + repeated greetings (redundancy the dedup pass catches).
    for i in range(4):
        items.append({"role": "user", "content":
            "Hi, thanks for the help earlier, just following up on my issue again."})
        items.append({"role": "assistant", "content":
            "Of course! Happy to help you follow up on your issue. Let me pull "
            "up your account details now, one moment please."})

    # Big wall of low-salience tool output BEFORE the fact.
    for i in range(blocks_per_side):
        items.append(_filler_block(i))

    # A retrieved doc that's mostly irrelevant policy boilerplate.
    items.append({"role": "tool", "kind": "doc", "content":
        "REFUND POLICY v7 — " + ("Refunds are processed within 5-7 business days. "
        "Refunds follow the original payment method unless otherwise required. ") * 60})

    # >>> The buried critical fact — sits in the dead center of the window.
    items.append({"role": "assistant", "kind": "fact", "content": CRITICAL})

    # Equally large wall of filler AFTER the fact (keeps it mid-context).
    for i in range(blocks_per_side):
        items.append(_filler_block(100 + i))
    for i in range(3):
        items.append({"role": "user", "content":
            "Also unrelated — do you know if your mobile app supports dark mode? "
            "And what about widget support on iOS 18? Just curious."})
        items.append({"role": "assistant", "content":
            "Great questions! Yes, dark mode is supported in the latest app, and "
            "home-screen widgets shipped recently. Now, back to your refund..."})

    # Near-duplicate of the policy doc (redundancy the dedup pass should catch).
    items.append({"role": "tool", "kind": "doc", "content":
        "REFUND POLICY v7 — " + ("Refunds are processed within 5-7 business days. "
        "Refunds follow the original payment method unless otherwise required. ") * 60})

    # The live task at the very end.
    items.append({"role": "user", "content":
        "Okay please go ahead and process my refund now to the right card."})

    return {
        "task": "Process the refund to the correct card for account ACC-99213.",
        "model": "claude-opus-4-8",
        "items": items,
    }


def build_suite(trace: dict) -> dict:
    return {
        "name": "contextforge-sample-suite",
        "description": "Buried-fact recall under a long, tangled support session.",
        "cases": [
            {
                "id": "buried-refund-card",
                "trace": trace,
                "question": "Which card must the refund go to, and which card must "
                            "NOT be used? Include the account ID.",
                # Distinctive phrases — avoid bare "4417", which collides with
                # random digits in the filler logs.
                "expected": "ending 4417",
                "keywords": ["ending 4417", "ACC-99213"],
            }
        ],
    }


def main() -> None:
    trace = build_trace()
    os.makedirs(os.path.join(ROOT, "examples"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "bench", "datasets"), exist_ok=True)

    with open(os.path.join(ROOT, "examples", "sample_trace.json"), "w") as f:
        json.dump(trace, f, indent=2)
    with open(os.path.join(ROOT, "bench", "datasets", "sample_suite.json"), "w") as f:
        json.dump(build_suite(trace), f, indent=2)

    print("wrote examples/sample_trace.json and bench/datasets/sample_suite.json")


if __name__ == "__main__":
    main()
