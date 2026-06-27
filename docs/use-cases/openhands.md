# ContextForge × OpenHands

**Keep an autonomous coding agent on-task across a long run.**

[OpenHands](https://github.com/All-Hands-AI/OpenHands) is an autonomous software
agent — it reads files, runs tests, edits code, and iterates for dozens of steps.
Every step appends more to the context: file contents, `pytest` output, stack
traces, git diffs. By step 40 the window is enormous, and an early constraint the
agent was given ("don't touch the frozen legacy billing module") is buried in the
dead center, where attention is weakest. So it edits the frozen module — and
breaks prod.

## What ContextForge does

It compiles each step's context before the model call: drops redundant test
output, trims stale logs, lifts the project constraints to the window **edges**,
and budgets the rest. The agent keeps its working memory of *what matters* without
drowning in its own tool output.

### Measured on a synthetic coding-agent session

[`examples/openhands_demo.py`](../../examples/openhands_demo.py) builds a long run
(coding-agent persona, repo memory, walls of `pytest` output, **a frozen-module
constraint buried mid-run**, then the live bug-fix task):

```
tokens      171,173 ->     19,655   (~88% smaller)
rot risk         33 ->         15   (moderate -> low)
items    114 actions applied

frozen-module constraint still visible to the model:   YES ✓
…and anchored to the window edge (not buried mid-run): YES ✓
```

```bash
python examples/openhands_demo.py
```

## Integration

Drop-in proxy — no changes inside OpenHands:

```bash
contextforge proxy --api anthropic --budget 24000      # or --api openai
```

Point OpenHands' LLM endpoint at it (e.g. `LLM_BASE_URL` / `OPENAI_API_BASE` =
`http://localhost:8788`). Every step is compiled before the model call; rot and
token deltas come back in `x-contextforge-*` response headers.

Prefer the SDK route (compile the context yourself) when you want to `pin`
constraints explicitly — see the [OpenClaw use case](openclaw.md#integration) for
the pattern.

## Caveats

The numbers are from a synthetic run for legibility — measure on your own OpenHands
traces. v0 proxy is non-streaming; SDK mode gives the strongest anchoring.
