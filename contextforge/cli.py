"""ContextForge command line interface.

    contextforge score   <trace.json>                 # rot risk + breakdown
    contextforge compile <trace.json> [--budget N]    # compress/reorder + report
    contextforge bench   <suite.json> [--model ...]   # measured accuracy delta
    contextforge demo                                 # run on the bundled sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .compiler import ContextCompiler, Policy
from .profiles import get_profile
from .rot import rot_score
from .salience import score_salience
from .tokens import using_accurate_tokenizer
from .types import CompileResult, Trace


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


_LEVEL_COLOR = {"low": "32", "moderate": "33", "high": "31", "severe": "1;31"}


def _bar(value: float, width: int = 24) -> str:
    filled = int(round((value / 100.0) * width))
    return "█" * filled + "·" * (width - filled)


def _print_rot(report, title="rot risk"):
    color = _LEVEL_COLOR.get(report.level, "0")
    print(f"  {title}: " + _c(f"{report.total:.0f}/100 ({report.level})", color))
    for name, val in report.components.items():
        print(f"    {name:<14} {_bar(val)} {val:>5.1f}")
    for note in report.notes:
        print("    " + _c("• " + note, "33"))


def cmd_score(args) -> int:
    trace = Trace.load(args.trace)
    model = args.model or trace.model
    profile = get_profile(model, args.profiles)
    score_salience(trace.items, trace.task)
    report = rot_score(trace.items, trace.task, model, **profile.rot_kwargs())
    print(_c("ContextForge — rot score", "1"))
    print(f"  trace: {args.trace}")
    print(f"  model: {model or 'unknown'}   profile: {profile.name}"
          f" (knee {profile.danger_start:,}–{profile.danger_full:,})")
    print(f"  items: {report.n_items}    tokens: {report.tokens:,}")
    print()
    _print_rot(report)
    if not using_accurate_tokenizer():
        print(_c("\n  (heuristic token counts — `pip install contextforge[tokens]`"
                 " for exact)", "2"))
    if args.json:
        print("\n" + json.dumps(report.to_dict(), indent=2))
    return 0


def _print_result(result: CompileResult):
    saved = result.tokens_saved
    print(_c("ContextForge — compile", "1"))
    print(
        f"  tokens   {result.tokens_before:>10,}  ->  {result.tokens_after:>10,}   "
        + _c(f"{result.savings_pct:+.1f}%  ({saved:+,} tok)",
             "32" if saved > 0 else "0")
    )
    print()
    _print_rot(result.rot_before, "rot before")
    print()
    _print_rot(result.rot_after, "rot after ")
    print()
    # action rollup
    rollup = {}
    for a in result.actions:
        rollup.setdefault(a.type, [0, 0])
        rollup[a.type][0] += 1
        rollup[a.type][1] += a.tokens_saved
    print(_c("  actions:", "1"))
    for t, (count, tok) in sorted(rollup.items()):
        sign = f"{tok:+,} tok" if tok else "context-add"
        print(f"    {t:<12} x{count:<4} {sign}")


def cmd_compile(args) -> int:
    trace = Trace.load(args.trace)
    policy = Policy(
        target_tokens=args.budget,
        dedup=not args.no_dedup,
        truncate=not args.no_truncate,
        reorder=not args.no_reorder,
        anchor=not args.no_anchor,
    )
    policy.apply_profile(get_profile(args.model or trace.model, args.profiles))
    result = ContextCompiler(policy=policy).compile(trace)
    _print_result(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(_c(f"\n  wrote compiled context -> {args.out}", "2"))
    if args.json:
        print("\n" + json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_bench(args) -> int:
    # The benchmark harness lives alongside the package in the repo (not shipped
    # as an installed module). Put the repo root on the path so it imports both
    # from a source checkout and via the installed console script.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from bench.benchmark import run_cli
    except ModuleNotFoundError:
        print("benchmark harness not found — run from a ContextForge source "
              "checkout (the `bench/` directory).", file=sys.stderr)
        return 1
    return run_cli(args)


def cmd_proxy(args) -> int:
    from .proxy import serve
    serve(api=args.api, port=args.port, budget=args.budget,
          upstream=args.upstream, dry_run=args.dry_run, profiles_path=args.profiles)
    return 0


def cmd_ui(args) -> int:
    from .server import serve
    serve(port=args.port, open_browser=args.open)
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import fit_profile
    from .profiles import save_profile

    with open(args.measurements, "r", encoding="utf-8") as f:
        data = json.load(f)
    measurements = data.get("measurements", data) if isinstance(data, dict) else data
    model = args.model or (data.get("model") if isinstance(data, dict) else None)
    if not model:
        print("error: provide --model or a 'model' field in the measurements file",
              file=sys.stderr)
        return 1

    profile = fit_profile(model, measurements)
    print(_c(f"ContextForge — calibrated profile for {model}", "1"))
    print(f"  measurements: {profile.n_samples}   fit R²: {profile.fit_r2}")
    print(f"  danger_start: {profile.danger_start:,} tokens")
    print(f"  danger_full:  {profile.danger_full:,} tokens")
    if args.save:
        path = save_profile(profile, args.profiles)
        print(_c(f"  saved -> {path}", "32"))
    else:
        print(_c("  (not saved; pass --save to add to the profile registry)", "2"))
    if args.json:
        print("\n" + json.dumps(profile.to_dict(), indent=2))
    return 0


def cmd_demo(args) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(here, "examples", "sample_trace.json")
    if not os.path.exists(sample):
        print("sample trace not found at", sample, file=sys.stderr)
        return 1
    print(_c(f"Running ContextForge on {sample}\n", "2"))
    trace = Trace.load(sample)
    result = ContextCompiler(target_tokens=args.budget).compile(trace)
    _print_result(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contextforge",
        description="A context compiler for LLM agents — stop context rot.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="report context-rot risk for a trace")
    s.add_argument("trace")
    s.add_argument("--model", default=None, help="model id (selects rot profile)")
    s.add_argument("--profiles", default=None, help="path to profile registry JSON")
    s.add_argument("--json", action="store_true", help="also emit raw JSON")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compile", help="compress/reorder a trace and report deltas")
    c.add_argument("trace")
    c.add_argument("--budget", type=int, default=None, help="target token budget")
    c.add_argument("--model", default=None, help="model id (selects rot profile)")
    c.add_argument("--profiles", default=None, help="path to profile registry JSON")
    c.add_argument("--out", help="write compiled context JSON here")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-dedup", action="store_true")
    c.add_argument("--no-truncate", action="store_true")
    c.add_argument("--no-reorder", action="store_true")
    c.add_argument("--no-anchor", action="store_true")
    c.set_defaults(func=cmd_compile)

    b = sub.add_parser("bench", help="measure accuracy + token delta on a suite")
    b.add_argument("suite")
    b.add_argument("--model", default=None, help="anthropic model id, or 'stub'")
    b.add_argument("--budget", type=int, default=None)
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("demo", help="run on the bundled sample trace")
    d.add_argument("--budget", type=int, default=4000)
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("proxy", help="run the drop-in compiling proxy server")
    pr.add_argument("--api", choices=["anthropic", "openai"], default="anthropic")
    pr.add_argument("--port", type=int, default=8788)
    pr.add_argument("--budget", type=int, default=None, help="target token budget")
    pr.add_argument("--upstream", default=None, help="override upstream base URL")
    pr.add_argument("--profiles", default=None, help="path to profile registry JSON")
    pr.add_argument("--dry-run", action="store_true",
                    help="return the compiled request instead of forwarding")
    pr.set_defaults(func=cmd_proxy)

    ui = sub.add_parser("ui", help="launch the web dashboard ('what the model saw')")
    ui.add_argument("--port", type=int, default=8799)
    ui.add_argument("--open", action="store_true", help="open a browser window")
    ui.set_defaults(func=cmd_ui)

    cal = sub.add_parser("calibrate", help="fit a per-model rot profile from sweep data")
    cal.add_argument("measurements", help="JSON from `python -m bench.sweep`")
    cal.add_argument("--model", default=None, help="model id for the profile")
    cal.add_argument("--profiles", default=None, help="registry path to save into")
    cal.add_argument("--save", action="store_true", help="save to the profile registry")
    cal.add_argument("--json", action="store_true")
    cal.set_defaults(func=cmd_calibrate)

    return p


def main(argv=None) -> int:
    from .env import load_dotenv
    load_dotenv()  # pick up ANTHROPIC_API_KEY from .env.local if present
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
