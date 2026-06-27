# ContextForge × support agent

**Stop a support agent from quietly going wrong after a long chat.**

A customer-support agent answers correctly for the first 30 turns, then starts
getting things wrong. Nobody changed the prompt. The chat just got long — order
lookups, KB articles, back-and-forth — and the one load-bearing fact (which card a
refund must go to) rotted in the middle of the window.

## What ContextForge does

It compiles the conversation each turn: dedups repeated KB/article pulls, trims
stale tool output, lifts the load-bearing account facts to the window **edges**,
and budgets the rest — so the model answers as if the chat were short and clean,
at a fraction of the tokens.

### Measured on the bundled support trace

[`examples/support_agent_demo.py`](../../examples/support_agent_demo.py) compiles
the bundled long support session (a refund-routing fact buried under ~250k tokens
of noise):

```
tokens      251,933 ->     19,999   (~92% smaller)
rot risk         44 ->         13   (moderate -> low)

buried refund fact (card 4417 / account ACC-99213) still visible: YES ✓
```

```bash
python examples/support_agent_demo.py
```

This is the same case as the bundled benchmark suite, so you can measure the
**accuracy** delta (not just tokens) against a real model:

```bash
contextforge bench bench/datasets/sample_suite.json --model claude-opus-4-8
```

On the offline proxy model: baseline misses the buried fact (`0.00`), compiled
recovers it (`1.00`), at ~89% fewer tokens.

## Integration

Drop-in proxy in front of your support agent's model
(`contextforge proxy --api anthropic --budget 20000`, then point the client's
`base_url` at it), or compile in code via the SDK and `pin` durable account facts.
See the [OpenClaw use case](openclaw.md#integration) for both patterns.
