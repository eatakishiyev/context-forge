"""Tests for the proxy adapters and the calibration/profiles modules."""

from __future__ import annotations

import os
import tempfile

from contextforge.adapters import (
    anthropic_to_trace, result_to_anthropic,
    openai_to_trace, result_to_openai, text_of,
)
from contextforge.calibrate import fit_knee, fit_profile
from contextforge.compiler import ContextCompiler, Policy
from contextforge.profiles import (
    ModelProfile, save_profile, load_registry, get_profile, DEFAULT_PROFILE,
)


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def test_text_of_flattens_blocks():
    assert text_of("hi") == "hi"
    assert text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert text_of([{"type": "tool_result", "content": "out"}]) == "out"


def _anthropic_body():
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 256,
        "system": "be precise about billing",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
            {"role": "user", "content": "the refund must go to the card ending 4417 "
                                        "for account ACC-1; never the card ending 2200."},
            {"role": "assistant", "content": "noted."},
            {"role": "user", "content": "process my refund to the right card now"},
        ],
    }


def test_anthropic_roundtrip_preserves_format_and_fact():
    body = _anthropic_body()
    trace = anthropic_to_trace(body)
    assert trace.task and "refund" in trace.task
    result = ContextCompiler(policy=Policy(target_tokens=2000)).compile(trace)
    new = result_to_anthropic(result, body)
    # Anthropic constraints: system is a string, messages start with user.
    assert isinstance(new["system"], str)
    assert new["messages"][0]["role"] == "user"
    assert new["stream"] is False
    # alternation: no two consecutive same-role messages
    roles = [m["role"] for m in new["messages"]]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    # the load-bearing fact survives somewhere
    blob = new["system"] + " " + " ".join(m["content"] for m in new["messages"])
    assert "4417" in blob and "ACC-1" in blob


def test_openai_roundtrip_preserves_roles():
    body = {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "the code is ALPHA-7, remember it"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "what is the code"},
        ],
    }
    trace = openai_to_trace(body)
    result = ContextCompiler(policy=Policy(target_tokens=2000)).compile(trace)
    new = result_to_openai(result, body)
    assert new["stream"] is False
    assert all(m["role"] in ("system", "user", "assistant") for m in new["messages"])
    blob = " ".join(m["content"] for m in new["messages"])
    assert "ALPHA-7" in blob


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_fit_knee_finds_degradation_bracket():
    # accuracy holds, then collapses between 150k and 250k tokens
    measurements = [
        (5_000, 1.0), (50_000, 1.0), (120_000, 1.0),
        (180_000, 0.6), (250_000, 0.0), (400_000, 0.0),
    ]
    start, full, r2 = fit_knee(measurements)
    assert 100_000 <= start <= 200_000
    assert start < full <= 260_000
    assert 0.0 <= r2 <= 1.0


def test_fit_knee_no_degradation_pushes_knee_out():
    measurements = [(5_000, 1.0), (100_000, 1.0), (500_000, 1.0)]
    start, full, _ = fit_knee(measurements)
    assert start >= 500_000 and full > start


def test_fit_profile_carries_metadata():
    prof = fit_profile("m1", [(1_000, 1.0), (100_000, 0.5), (200_000, 0.0)])
    assert prof.name == "m1" and prof.n_samples == 3
    assert prof.danger_start < prof.danger_full


# --------------------------------------------------------------------------- #
# profiles registry
# --------------------------------------------------------------------------- #
def test_profile_save_and_get_with_prefix_match():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "profiles.json")
        save_profile(ModelProfile(name="claude-opus", danger_start=120_000,
                                  danger_full=200_000), path)
        reg = load_registry(path)
        assert "claude-opus" in reg
        # prefix match: a versioned id resolves to the family profile
        prof = get_profile("claude-opus-4-8", path)
        assert prof.danger_start == 120_000


def test_get_profile_falls_back_to_default():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "profiles.json")
        assert get_profile("unknown-model", path).name == DEFAULT_PROFILE.name


def test_policy_apply_profile():
    p = Policy()
    p.apply_profile(ModelProfile(name="x", danger_start=10, danger_full=20))
    assert p.danger_start == 10 and p.danger_full == 20
    assert p.rot_kwargs()["danger_start"] == 10
