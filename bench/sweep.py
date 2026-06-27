"""Calibration sweep — generate accuracy-vs-context-size measurements.

Runs a model on the *uncompiled* buried-fact task at a range of context sizes and
records (tokens, accuracy). The resulting curve is what ``contextforge calibrate``
fits a per-model rot knee to.

    python -m bench.sweep --model stub --out profiles/stub_measurements.json
    python -m bench.sweep --model claude-opus-4-8 --out profiles/opus_measurements.json

With the stub runner this is deterministic and offline (it demonstrates the
pipeline). With a real model it produces a genuine degradation curve for that
model — the input to a real calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextforge.calibrate import fit_profile  # noqa: E402
from contextforge.tokens import count_items_tokens  # noqa: E402
from contextforge.types import Trace  # noqa: E402
from bench.benchmark import make_runner, score_answer  # noqa: E402
from bench.make_sample import build_trace, CRITICAL  # noqa: E402

# Block counts per side -> roughly spans a few k up to ~400k tokens.
DEFAULT_STEPS = [0, 2, 6, 12, 20, 30, 45, 65]

QUESTION = ("Which card must the refund go to, and which card must NOT be used? "
            "Include the account ID.")
EXPECTED = "ending 4417"
KEYWORDS = ["ending 4417", "ACC-99213"]


def run_sweep(model: str, steps=None):
    steps = steps or DEFAULT_STEPS
    runner = make_runner(model)
    measurements = []
    for blocks in steps:
        trace = Trace.from_dict(build_trace(blocks_per_side=blocks))
        tokens = count_items_tokens(trace.items, trace.model)
        msgs = [{"role": i.role, "content": i.content} for i in trace.items]
        ans = runner.answer(msgs, QUESTION)
        acc = score_answer(ans, EXPECTED, KEYWORDS)
        measurements.append([tokens, acc])
        print(f"  blocks/side={blocks:>3}  tokens={tokens:>9,}  accuracy={acc:.2f}")
    return runner.name, measurements


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ContextForge calibration sweep")
    p.add_argument("--model", default="stub")
    p.add_argument("--out", default=None, help="write measurements JSON here")
    args = p.parse_args(argv)

    print(f"Sweeping context sizes with model={args.model} ...")
    name, measurements = run_sweep(args.model)
    profile = fit_profile(name, measurements)

    print("\nFitted knee:")
    print(f"  danger_start = {profile.danger_start:,} tokens")
    print(f"  danger_full  = {profile.danger_full:,} tokens   (R²={profile.fit_r2})")

    out = {"model": name, "measurements": measurements,
           "fitted_profile": profile.to_dict()}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
