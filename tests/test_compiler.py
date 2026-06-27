"""Tests for the ContextForge core. Run: pytest -q"""

from __future__ import annotations

import copy

import pytest

from contextforge import ContextCompiler, Policy, Trace, ContextItem, rot_score
from contextforge.salience import score_salience
from contextforge.tokens import count_tokens, count_items_tokens
from contextforge.compress import dedup_items, truncate_stale
from contextforge.reorder import extract_salient_facts, edge_anchor
from contextforge.budget import enforce_budget


def _items(*pairs):
    return [ContextItem(content=c, role=r, id=f"i{idx}")
            for idx, (r, c) in enumerate(pairs)]


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #
def test_token_count_monotonic():
    assert count_tokens("hello world") > 0
    assert count_tokens("a" * 400) > count_tokens("a" * 40)


# --------------------------------------------------------------------------- #
# rot scoring
# --------------------------------------------------------------------------- #
def test_rot_load_scales_with_tokens():
    small = [ContextItem(content="x " * 100)]
    big = [ContextItem(content="x " * 200_000)]
    assert rot_score(big).components["load"] > rot_score(small).components["load"]


def test_rot_redundancy_detects_duplicates():
    dupes = _items(*[("user", "the deployment failed because of a timeout")] * 8)
    varied = _items(
        ("user", "the cat sat on the mat"),
        ("user", "quarterly revenue grew twelve percent"),
        ("user", "the bridge spans four hundred meters"),
        ("user", "photosynthesis converts light to sugar"),
        ("user", "the election is scheduled for november"),
        ("user", "her flight departs at dawn tomorrow"),
        ("user", "the recipe calls for two cups flour"),
        ("user", "mount everest is the tallest peak"),
    )
    assert rot_score(dupes).components["redundancy"] > \
        rot_score(varied).components["redundancy"]


def test_rot_level_thresholds():
    assert rot_score([ContextItem(content="hi")]).level == "low"


# --------------------------------------------------------------------------- #
# salience
# --------------------------------------------------------------------------- #
def test_salience_pinned_is_max():
    items = [ContextItem(content="anything", pinned=True)]
    score_salience(items, task="unrelated query")
    assert items[0].salience == 1.0


def test_salience_rewards_task_relevance():
    items = _items(
        ("user", "the refund must go to card ending 4417 for the account"),
        ("user", "by the way the weather is lovely and the coffee was great"),
    )
    score_salience(items, task="which card gets the refund")
    assert items[0].salience > items[1].salience


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def test_dedup_removes_near_duplicates_keeps_one():
    items = _items(
        ("tool", "ERROR: connection refused at host db-01 port 5432 retry now"),
        ("tool", "ERROR: connection refused at host db-01 port 5432 retry now."),
        ("user", "completely different unique content here about giraffes"),
    )
    kept, actions = dedup_items(items, threshold=0.9)
    assert len(kept) == 2
    assert any(a.type == "dedup" for a in actions)


def test_dedup_never_drops_pinned():
    items = [
        ContextItem(content="same text here", pinned=True, id="a"),
        ContextItem(content="same text here", id="b"),
    ]
    kept, _ = dedup_items(items, threshold=0.5)
    assert any(i.id == "a" for i in kept)


def test_truncate_clips_stale_lowsalience():
    long_low = ContextItem(content="filler " * 500, role="tool", salience=0.1, id="x")
    recent = ContextItem(content="recent important", role="user", salience=0.9, id="y")
    items = [long_low] + [ContextItem(content=f"pad {i}", id=f"p{i}") for i in range(8)] + [recent]
    before = count_tokens(items[0].content)
    truncate_stale(items, keep_recent=6)
    assert count_tokens(items[0].content) < before
    assert "elided" in items[0].content


# --------------------------------------------------------------------------- #
# reorder / anchoring
# --------------------------------------------------------------------------- #
def test_extract_salient_facts_finds_key_statement():
    items = _items(
        ("system", "you are a helpful assistant"),
        ("assistant", "The account ID is ACC-99213 and the refund is $428.50."),
    )
    score_salience(items, task="what is the account id and refund")
    facts = extract_salient_facts(items, task="account id refund")
    assert any("ACC-99213" in f for f in facts)


def test_edge_anchor_places_facts_at_both_edges():
    items = _items(
        ("system", "be precise"),
        ("assistant", "The secret code is ALPHA-7 and must be remembered."),
        ("user", "ok"),
    )
    items[0].pinned = True
    score_salience(items, task="what is the secret code")
    out, actions = edge_anchor(items, task="secret code")
    assert out[0].pinned and out[-1].meta.get("cf_generated")
    assert any("ALPHA-7" in out[i].content for i in (0, 1))  # near the front
    assert "ALPHA-7" in out[-1].content                       # and at the back


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def test_budget_enforced_and_drops_lowest_salience():
    items = [
        ContextItem(content="word " * 200, salience=0.1, id="low"),
        ContextItem(content="word " * 200, salience=0.9, id="high"),
    ]
    # Budget fits exactly one of the two ~254-token items.
    kept, actions = enforce_budget(items, target_tokens=300)
    assert count_items_tokens(kept) <= 300
    kept_ids = {i.id for i in kept}
    assert "high" in kept_ids and "low" not in kept_ids  # low salience dropped first


def test_budget_respects_pin():
    items = [ContextItem(content="x " * 500, pinned=True, salience=0.0, id="p")]
    kept, _ = enforce_budget(items, target_tokens=10)
    assert any(i.id == "p" for i in kept)


# --------------------------------------------------------------------------- #
# compiler end-to-end
# --------------------------------------------------------------------------- #
def _build_buried_trace():
    items = [ContextItem(content="system rules", role="system", pinned=True)]
    items += [ContextItem(content=f"log line noise number {i} " * 30, role="tool",
                          kind="tool_result") for i in range(20)]
    items.append(ContextItem(content="CRITICAL: the launch date is 2026-09-01.",
                             role="assistant", kind="fact"))
    items += [ContextItem(content=f"more noise {i} " * 30, role="tool",
                          kind="tool_result") for i in range(20)]
    items.append(ContextItem(content="when is the launch?", role="user"))
    return Trace(items=items, task="when is the launch date", model=None)


def test_compile_reduces_tokens_and_rot():
    trace = _build_buried_trace()
    result = ContextCompiler(target_tokens=2000).compile(trace)
    assert result.tokens_after < result.tokens_before
    assert result.rot_after.total <= result.rot_before.total


def test_compile_preserves_critical_fact():
    trace = _build_buried_trace()
    result = ContextCompiler(target_tokens=2000).compile(trace)
    blob = "\n".join(i.content for i in result.items)
    assert "2026-09-01" in blob


def test_compile_does_not_mutate_input():
    trace = _build_buried_trace()
    snapshot = copy.deepcopy(trace.to_dict())
    ContextCompiler(target_tokens=2000).compile(trace)
    assert trace.to_dict() == snapshot


def test_compile_respects_budget_ceiling():
    trace = _build_buried_trace()
    result = ContextCompiler(target_tokens=1500).compile(trace)
    assert result.tokens_after <= 1500 * 1.15  # anchors add a little overhead


def test_policy_toggles_disable_passes():
    trace = _build_buried_trace()
    p = Policy(dedup=False, truncate=False, reorder=False, anchor=False, budget=False)
    result = ContextCompiler(policy=p).compile(trace)
    assert all(a.type == "normalize" for a in result.actions) or not result.actions


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
