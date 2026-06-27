# ContextForge benchmark

The reproducible context-rot benchmark — the magnet artifact. Point it at traces,
get a **measured accuracy delta + token delta** for raw vs. ContextForge-compiled
context.

## Run it

```bash
# zero-setup proxy model — deterministic, no API key, CI-safe
python -m bench.benchmark bench/datasets/sample_suite.json --model stub

# real frontier model
export ANTHROPIC_API_KEY=...
python -m bench.benchmark bench/datasets/sample_suite.json --model claude-opus-4-8
```

## Suite format

A suite is JSON: a list of `cases`, each pairing a long-context trace with a
question and the expected answer.

```json
{
  "name": "my-suite",
  "cases": [
    {
      "id": "buried-refund-card",
      "trace": { "task": "...", "model": "...", "items": [ ... ] },
      "question": "Which card must the refund go to?",
      "expected": "4417",
      "keywords": ["4417", "ACC-99213"]
    }
  ]
}
```

Scoring: an answer is correct (1.0) if it contains `expected` (case-insensitive)
**or** all `keywords`; otherwise it earns partial credit for the fraction of
keywords present. Bring your own scorer for richer evals.

## Model runners

| Runner | Needs | Use |
|---|---|---|
| `stub` | nothing | CI / demos. A deterministic proxy that emulates context rot via **U-shaped positional attention** (edges strong, middle weak) discounted by total length. Shows the *shape* of the result; **not** a substitute for a real model. |
| `claude-opus-4-8` (or any Anthropic id) | `ANTHROPIC_API_KEY`, `pip install contextforge[anthropic]` | Real measured accuracy. |

Add your own by implementing `.answer(messages, question) -> str` (see
`AnthropicRunner`).

## Generating the sample

`examples/sample_trace.json` and `bench/datasets/sample_suite.json` are generated
reproducibly (no randomness):

```bash
python -m bench.make_sample
# control size: ~250k tokens by default
CF_SAMPLE_BLOCKS=40 python -m bench.make_sample
```

The sample is a long, tangled support session with a **load-bearing fact buried
in the dead center**, heavy low-salience tool output, and a duplicate policy doc —
the canonical context-rot setup. It is filler-heavy *by design* so the
degradation is legible; real traces vary.

## Calibration sweep

`sweep.py` measures accuracy vs. context size (uncompiled) to produce the
degradation curve that `contextforge calibrate` fits a per-model knee to:

```bash
python -m bench.sweep --model stub --out profiles/stub.json      # offline demo
python -m bench.sweep --model claude-opus-4-8 --out profiles/opus.json
contextforge calibrate profiles/opus.json --model claude-opus-4-8 --save
```

With the stub the curve is deterministic (recall holds, then collapses past
~200k tokens), demonstrating the full sweep → fit → profile pipeline. With a real
model it yields a genuine knee for that model.

## What "good" looks like

On the bundled suite with `--model stub`: baseline accuracy **0.00** (the buried
fact is lost mid-context), compiled accuracy **1.00** (anchored to the edge),
~**89%** token savings. The aim of the 30-day test is to reproduce that *direction*
on real teams' traces with a real model.
