# ContextForge — Strategy

*A context compiler that prevents context rot. Compiler + linter for context, not
another vector store.*

---

## 1. Thesis

Every frontier model degrades as input grows — **before** the advertised window
limit, with a sharp knee well under the "1M-token" ceiling, and worst inside long
multi-tool agent sessions. The failure mode was finally **named and benchmarked**
(Chroma's context-rot work, 2025) but remains **unowned** by any product.

ContextForge owns it: a drop-in layer that scores, compresses, reorders, and
budgets context so the model performs as if the input were short and clean —
recovering both **quality** and **cost** in one move.

## 2. Why now

Two unlocks collided in ~12 months:

1. **Agents run for hours/days** and accumulate enormous contexts.
2. **The failure mode became legible** — named, measured, reproducible.

The problem is now acutely felt by every team shipping agents, and there is no
product systematically preventing it. That gap is the opening.

## 3. Wedge → platform

| Stage | What it is | Why it works |
|---|---|---|
| **Wedge** | OSS SDK/proxy that trims + reorders context and emits a *rot score* per call | Zero-friction adoption; the score is a shareable, CI-able number |
| **+ Benchmark** | A reproducible, open context-rot benchmark | Distribution magnet (HN/X); establishes us as the authority on the problem |
| **Platform** | Managed context layer: eval-driven compression policies, per-model tuning, "what the model saw" observability dashboard | Recurring value + switching cost once teams gate CI on the rot score |

The benchmark is top-of-funnel; the compiler is the conversion; the managed layer
is the revenue.

## 4. The single highest-leverage artifact

The **30-day smallest test** is simultaneously the proof, the marketing, and the
product seed:

> Publish an open context-rot benchmark **and** a CLI that, given a real
> long-context trace, returns a compressed/re-ordered context with a **measured
> accuracy + token delta**.
>
> **Success = 20+ teams run it on their own traces and report a meaningful
> accuracy or cost win.**

This repo is that artifact. Everything else (dashboard, managed service, per-model
policy tuning) is downstream.

## 5. Business model & pricing

- **Open-source core** (Apache-2.0) — adoption flywheel.
- **Usage-based managed service** — priced per **million tokens processed**; token
  savings fund the ROI story (the service pays for itself).
- **Team tier ($99–$499/mo)** — policies, analytics, per-model tuning, the
  observability dashboard.

## 6. Go-to-market

Classic dev-infra wedge:

1. Open-source the compiler.
2. Publish the reproducible context-rot benchmark (HN/X magnet).
3. Integrate with LangChain / LlamaIndex / major agent frameworks so we ride
   their adoption — treated as *distribution*, not as the moat (frameworks churn).
4. Land-and-expand: free CLI → CI gating on rot score → managed layer.

## 7. Competition (adjacent, not direct)

| Category | Examples | Why they don't cover this |
|---|---|---|
| RAG tooling | LlamaIndex | Retrieval, not in-window degradation |
| Memory layers | Mem0, Zep, Letta | Store/recall state; don't score or remediate rot per call |
| Prompt-eval platforms | (various) | Measure prompts; no automatic context remediation |

**Whitespace:** nobody owns context rot as a *first-class, measurable failure with
an automatic remediation layer.* We own the **benchmark and the fix together**.

## 8. Moat

Not any single model call. It is:

1. A **proprietary corpus** of context-rot evals across models and tasks.
2. **Learned compression/ordering policies** tuned per model.
3. **Switching cost** once teams trust the rot score in CI.

The eval data and policies compound; the API surface is replaceable.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Vendors close the long-context gap** (the main risk) | Frame around **cost AND quality** — token savings never stop mattering. Go deep on **agent sessions**, where rot persists longest. |
| **Lossy compression silently drops the critical fact** — our product *causing* a rot-style failure | v0 is **extractive + auditable**, never paraphrastic; critical items are **pinned**; everything is logged. Abstractive summarization is opt-in only. |
| **Cold-start on eval data** (no corpus on day 1) | Bootstrap from the open benchmark + opted-in user traces. |
| **Framework churn** | Treat integrations as distribution, not moat; invest in the CI/rot-score habit. |
| **Trust in the rot score** | The benchmark is how teams *calibrate the score against their own accuracy* — trust is earned with measured deltas, not asserted. |

## 10. Explosion score

**53.0 / 65** — original, datable, painful, and compounding (quality + cost in
one). Main risk is vendor improvement in long-context attention; mitigated by the
cost framing and the agent-session focus.

## 11. Roadmap from this repo

- [x] Core compiler: score · compress · reorder · budget (extractive, auditable)
- [x] Rot-risk score with component breakdown + CLI
- [x] Reproducible benchmark harness (stub + Anthropic runners)
- [x] Bundled buried-fact suite demonstrating baseline-fails / compiled-recovers
- [x] Per-model rot calibration — sweep → fit knee → profile registry → auto-applied
- [x] Drop-in proxy (Anthropic/OpenAI-compatible endpoint) with rot/token headers
- [ ] Abstractive summarizer plug-in (LLM-backed, behind a quality gate)
- [ ] Streaming support in the proxy + framework adapters (LangChain/LlamaIndex)
- [ ] Observability dashboard: "what the model actually saw" per call
- [ ] Expand the open benchmark across models, tasks, and real agent traces
- [ ] Seed the profile registry with real fitted knees per frontier model
