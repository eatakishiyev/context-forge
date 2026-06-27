# ContextForge × OpenClaw

**Keep a 24/7 personal agent sharp — and cheap — as its context balloons.**

[OpenClaw](https://openclaw.ai) is an always-on, locally-run personal AI agent:
persistent memory, 100+ connected services (Gmail, Calendar, GitHub, Slack,
Spotify, browser…), and autonomous task loops that run around the clock across
WhatsApp/Telegram/Discord/etc. It supports multiple model backends (Claude, GPT,
others).

That profile is the textbook setup for **context rot**.

## Why OpenClaw is a context-rot magnet

| OpenClaw trait | Effect on the context window |
|---|---|
| Sessions that never end (24/7) | Context grows without bound across days |
| Persistent memory + personalization | More state injected on every turn |
| 100+ integrations, autonomous loops | Walls of low-salience tool output (background syncs, polls) |
| Standing rules & user preferences | Load-bearing facts get **buried mid-context** under the noise |

The failure isn't a crash — it's silent. Around turn 200, the model quietly stops
honoring a standing rule the user set on turn 12 ("never schedule anything on
Fridays"), because that one line is now stranded in the dead center of a 200k-token
context, where attention is weakest. Cost climbs in lockstep: every turn re-sends
the whole bloated history.

## What ContextForge does about it

It sits between OpenClaw and the model and **compiles each turn's context**:
scores rot risk, removes redundant sync dumps, trims stale low-salience material,
lifts the user's standing rules to the **edges** of the window (where attention is
strongest), and enforces a token budget — emitting a rot score and token delta per
call.

### Measured on a synthetic OpenClaw session

The runnable example ([`examples/openclaw_proxy.py`](../../examples/openclaw_proxy.py))
builds a realistic long session — OpenClaw persona, persistent memory, a wall of
Gmail/Calendar/GitHub background-sync output, **one standing rule buried in the
middle** ("never schedule on Fridays"), then a live task ("book a dentist next
week"). Running the compiler on it:

```
tokens      228,085 ->     19,879   (91.3% smaller)
rot risk         40 ->         14   (moderate -> low)
items    94 actions applied

buried standing rule ('no Fridays') still visible to the model: YES ✓
…and lifted into an edge anchor (not stranded mid-context):     YES ✓
```

The model now sees a ~20k-token, edge-anchored context instead of 228k of tangled
history — the standing rule is back at the window edge, and the token bill drops
~91%.

```bash
python examples/openclaw_proxy.py
```

## Integration

### Option A — drop-in proxy (zero code changes in OpenClaw)

Run the compiling proxy next to OpenClaw and point OpenClaw's model provider at it:

```bash
contextforge proxy --api anthropic --budget 20000      # or --api openai
```

```bash
# in OpenClaw's model config / environment
ANTHROPIC_BASE_URL=http://localhost:8788
# or, for an OpenAI-backed OpenClaw model:
OPENAI_BASE_URL=http://localhost:8788
```

Every request OpenClaw sends is compiled before it reaches the model. Responses
carry observability headers:

```
x-contextforge-rot-before / x-contextforge-rot-after
x-contextforge-tokens-before / x-contextforge-tokens-after / x-contextforge-tokens-saved
```

Pipe those into your dashboards to watch rot and spend per turn.

### Option B — SDK (richer control)

If you can touch OpenClaw's model call, compile the context yourself and pass
metadata (`pinned`, `kind="memory"`/`"fact"`) so anchoring is even more precise:

```python
from contextforge import ContextCompiler, Trace

trace  = Trace.from_dict({"task": user_task, "model": "claude-opus-4-8",
                          "items": openclaw_messages})
result = ContextCompiler(target_tokens=20_000).compile(trace)

messages = result.to_messages()          # hand straight to the model SDK
log(result.rot_before.total, "→", result.rot_after.total, result.savings_pct)
```

Mark durable user rules/preferences with `pinned: true` so they are never dropped
and always anchored.

## Tuning for OpenClaw

- **Calibrate the rot knee per model** OpenClaw uses, so the score reflects real
  behavior: `contextforge calibrate …` (see the main README).
- **Budget** to whatever keeps cost/latency sane (20–40k is a good start for an
  agent turn); critical state survives via anchoring regardless of budget.
- **Pin standing rules.** Anything the user phrases as "always / never …" should be
  `pinned` (SDK) so it is guaranteed into the window.

## Caveats

- The numbers above are from a *synthetic* session designed to be legible; run the
  example on your own OpenClaw traces for real figures.
- The v0 proxy is non-streaming and flattens content blocks to text.
- From raw proxied messages (no `pinned`/`kind` metadata) anchoring is best-effort;
  Option B gives the strongest guarantees.
