"""ContextForge benchmark harness — the reproducible context-rot benchmark.

Given a *suite* (long-context traces, each with a question + expected answer), it
runs the model on (a) the raw context and (b) the ContextForge-compiled context,
and reports the accuracy delta and token delta. This is the magnet artifact: a
team points it at their own traces and gets a measured win (or doesn't).

Model runners are pluggable:
  * StubRunner       — no API key needed; deterministic keyword-recall proxy so
                       the harness runs anywhere and in CI. Demonstrates the
                       *shape* of the result; not a substitute for a real model.
  * AnthropicRunner  — real accuracy against a frontier model (needs
                       ANTHROPIC_API_KEY and `pip install contextforge[anthropic]`).

Usage:
    python -m bench.benchmark bench/datasets/sample_suite.json --model stub
    contextforge bench bench/datasets/sample_suite.json --model claude-opus-4-8
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextforge.compiler import ContextCompiler, Policy  # noqa: E402
from contextforge.textutil import keywords  # noqa: E402
from contextforge.types import Trace  # noqa: E402


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_answer(answer: str, expected: str, keywords_required: List[str]) -> float:
    """Return 1.0 if the answer is correct, else partial credit in [0,1].

    Correct = contains the expected string (case-insensitive) OR contains all
    required keywords. Partial credit = fraction of required keywords present.
    """
    a = (answer or "").lower()
    if expected and expected.lower() in a:
        return 1.0
    reqs = [k.lower() for k in (keywords_required or [])]
    if not reqs:
        return 0.0
    hits = sum(1 for k in reqs if k in a)
    if hits == len(reqs):
        return 1.0
    return hits / len(reqs)


# --------------------------------------------------------------------------- #
# Model runners
# --------------------------------------------------------------------------- #
class StubRunner:
    """API-free proxy model.

    Emulates context rot: it 'answers' by scanning the context for the question's
    keywords, but its recall *degrades* as the context grows and as relevant
    facts sit further from the edges — the exact dynamics ContextForge targets.
    So compiled (shorter, edge-anchored) contexts score higher, mirroring real
    model behavior in direction if not magnitude.
    """

    name = "stub"

    # A block is only "recalled" if the model's modeled attention to it clears
    # this bar. Tuned so edge items in a short context pass and mid items in a
    # long context fail — the context-rot signature.
    ATTN_THRESHOLD = 0.45

    def _attention(self, idx: int, n: int, total_tokens: int) -> float:
        """Modeled recall weight for a block — the context-rot signature.

        Two length-dependent effects, matching the empirical picture:
        * middle-dip: "lost in the middle" *worsens with length*. In a short
          context the middle is fine; as the context grows, central positions
          collapse while the edges hold.
        * global decay: a mild overall recall penalty as the context grows.
        """
        pos = idx / (n - 1) if n > 1 else 0.0
        u = 4.0 * (pos - 0.5) ** 2  # 0 at center, 1 at edges
        dip = min(1.0, total_tokens / 300_000.0)        # how deep the middle sags
        positional = 1.0 - 0.85 * dip * (1.0 - u)       # edges stay ~1.0
        global_decay = 1.0 / (1.0 + total_tokens / 600_000.0)
        return positional * global_decay

    def answer(self, messages: List[Dict[str, str]], question: str) -> str:
        qkw = keywords(question)
        blocks = [m["content"] for m in messages]
        n = len(blocks)
        # ~4 chars/token, matching contextforge.tokens heuristic.
        total_tokens = sum(len(b) for b in blocks) / 4.0

        # Gather candidate sentences from blocks the model actually attends to,
        # ranked by question overlap (the model recalls the most relevant span).
        cands = []  # (overlap_count, attn, sentence)
        for idx, block in enumerate(blocks):
            attn = self._attention(idx, n, total_tokens)
            if attn < self.ATTN_THRESHOLD:
                continue
            for sent in block.replace("\n", " ").split("."):
                ov = len(qkw & keywords(sent))
                if ov:
                    cands.append((ov, attn, sent.strip()))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        return " | ".join(s for _, _, s in cands[:4])


class AnthropicRunner:
    """Real model runner via the Anthropic SDK."""

    def __init__(self, model: str = "claude-opus-4-8"):
        self.name = model
        try:
            import anthropic  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK not installed — `pip install contextforge[anthropic]`"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY to use a real model runner")
        self._client = anthropic.Anthropic()
        self.model = model

    def answer(self, messages: List[Dict[str, str]], question: str) -> str:
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] in ("user", "assistant")]
        # collapse non-system context into a single user turn followed by the Q
        ctx = "\n\n".join(f"[{m['role']}] {m['content']}" for m in convo)
        user = f"{ctx}\n\nBased only on the context above, answer: {question}"
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system="\n\n".join(sys_parts) or "Answer concisely from the context.",
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def make_runner(model: Optional[str]):
    if not model or model == "stub":
        return StubRunner()
    from contextforge.env import load_dotenv
    load_dotenv()  # pick up ANTHROPIC_API_KEY from .env.local
    return AnthropicRunner(model)


# --------------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    id: str
    baseline_score: float
    compiled_score: float
    tokens_before: int
    tokens_after: int
    rot_before: float
    rot_after: float

    @property
    def accuracy_delta(self) -> float:
        return self.compiled_score - self.baseline_score

    @property
    def savings_pct(self) -> float:
        if not self.tokens_before:
            return 0.0
        return 100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before


def run_suite(suite: dict, runner, policy: Optional[Policy] = None) -> List[CaseResult]:
    compiler = ContextCompiler(policy=policy or Policy(target_tokens=None))
    results: List[CaseResult] = []
    for case in suite["cases"]:
        trace = Trace.from_dict(case["trace"])
        question = case["question"]
        expected = case.get("expected", "")
        req = case.get("keywords", [])

        baseline_msgs = [{"role": i.role, "content": i.content} for i in trace.items]
        compiled = compiler.compile(trace)
        compiled_msgs = compiled.to_messages()

        base_ans = runner.answer(baseline_msgs, question)
        comp_ans = runner.answer(compiled_msgs, question)

        results.append(
            CaseResult(
                id=case.get("id", "case"),
                baseline_score=score_answer(base_ans, expected, req),
                compiled_score=score_answer(comp_ans, expected, req),
                tokens_before=compiled.tokens_before,
                tokens_after=compiled.tokens_after,
                rot_before=compiled.rot_before.total,
                rot_after=compiled.rot_after.total,
            )
        )
    return results


def summarize(results: List[CaseResult], runner_name: str) -> dict:
    n = len(results) or 1
    base = sum(r.baseline_score for r in results) / n
    comp = sum(r.compiled_score for r in results) / n
    tok_before = sum(r.tokens_before for r in results)
    tok_after = sum(r.tokens_after for r in results)
    savings = 100.0 * (tok_before - tok_after) / tok_before if tok_before else 0.0
    return {
        "runner": runner_name,
        "n_cases": len(results),
        "baseline_accuracy": round(base, 3),
        "compiled_accuracy": round(comp, 3),
        "accuracy_delta": round(comp - base, 3),
        "tokens_before": tok_before,
        "tokens_after": tok_after,
        "token_savings_pct": round(savings, 1),
        "cases": [r.__dict__ for r in results],
    }


def run_cli(args) -> int:
    with open(args.suite, "r", encoding="utf-8") as f:
        suite = json.load(f)
    runner = make_runner(args.model)
    policy = Policy(target_tokens=getattr(args, "budget", None))
    results = run_suite(suite, runner, policy)
    summary = summarize(results, runner.name)

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
        return 0

    print(f"ContextForge benchmark — {suite.get('name', args.suite)}")
    print(f"  runner: {summary['runner']}   cases: {summary['n_cases']}")
    print("-" * 64)
    print(f"  {'case':<22}{'base':>7}{'comp':>7}{'Δacc':>7}{'tok save':>11}")
    for r in results:
        print(
            f"  {r.id:<22}{r.baseline_score:>7.2f}{r.compiled_score:>7.2f}"
            f"{r.accuracy_delta:>+7.2f}{r.savings_pct:>10.0f}%"
        )
    print("-" * 64)
    print(
        f"  {'OVERALL':<22}{summary['baseline_accuracy']:>7.2f}"
        f"{summary['compiled_accuracy']:>7.2f}{summary['accuracy_delta']:>+7.2f}"
        f"{summary['token_savings_pct']:>10.0f}%"
    )
    return 0


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="ContextForge context-rot benchmark")
    p.add_argument("suite")
    p.add_argument("--model", default="stub")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--json", action="store_true")
    return run_cli(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(_main())
