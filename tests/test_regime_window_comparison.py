"""Comparing two windows' readings of the same symbol.

The property that matters most is symmetry: no test below tells the comparison
which side is the longer history, because the method must not know.
"""
from __future__ import annotations

import pytest

from lazystats.regimes.tiers import calm_or_highvol, tier_of, volatility_tiers
from lazystats.regimes.window_comparison import WindowFit, compare_fits, summarise


def state(index: int, vol: float, mean: float = 0.0) -> dict:
    return {"state": index, "annualized_volatility": vol, "annualized_mean_return": mean}


def fit(window: str, *, current: int | None, states: list[dict]) -> WindowFit:
    return WindowFit(window=window, n_states=len(states), current_state=current,
                     states=tuple(states), as_of="2026-08-13", data_start="2010-01-01")


CALM_THEN_WILD = [state(0, 0.10, 0.05), state(1, 0.35, -0.20)]


class TestAgreement:
    def test_the_same_current_tier_agrees(self):
        a = fit("full", current=0, states=CALM_THEN_WILD)
        b = fit("8y", current=0, states=CALM_THEN_WILD)
        assert compare_fits(a, b)["agreement"] == "agree"

    def test_a_different_current_tier_disagrees(self):
        a = fit("full", current=0, states=CALM_THEN_WILD)
        b = fit("8y", current=1, states=CALM_THEN_WILD)
        assert compare_fits(a, b)["agreement"] == "disagree"

    def test_a_single_state_fit_is_neither(self):
        """One state has nothing to rank against; calling it agreement would
        assert something the fit never said."""
        a = fit("full", current=0, states=[state(0, 0.2)])
        b = fit("8y", current=0, states=CALM_THEN_WILD)
        assert compare_fits(a, b)["agreement"] == "single_state"


class TestSymmetry:
    """Neither window is privileged. This is the whole point of the rewrite."""

    def test_swapping_the_sides_preserves_the_verdict(self):
        a = fit("3y", current=0, states=CALM_THEN_WILD)
        b = fit("10y", current=1, states=CALM_THEN_WILD)
        assert compare_fits(a, b)["agreement"] == compare_fits(b, a)["agreement"]

    def test_swapping_the_sides_swaps_the_reported_windows(self):
        a = fit("3y", current=0, states=CALM_THEN_WILD)
        b = fit("10y", current=0, states=CALM_THEN_WILD)
        forward, backward = compare_fits(a, b), compare_fits(b, a)
        assert forward["baseline"]["window"] == backward["candidate"]["window"] == "3y"
        assert forward["candidate"]["window"] == backward["baseline"]["window"] == "10y"

    def test_two_bounded_windows_compare_like_any_other_pair(self):
        """Nothing requires one side to be the unrestricted history."""
        result = compare_fits(fit("3y", current=0, states=CALM_THEN_WILD),
                              fit("10y", current=1, states=CALM_THEN_WILD))
        assert result["status"] == "ok"
        assert result["agreement"] == "disagree"

    def test_the_method_never_names_a_window_itself(self):
        """A hardcoded '8y' or 'full' anywhere in the verdict would put a
        project's choice back inside the method."""
        result = compare_fits(fit("3y", current=0, states=CALM_THEN_WILD),
                              fit("10y", current=0, states=CALM_THEN_WILD))
        assert result["baseline"]["window"] == "3y"
        assert result["candidate"]["window"] == "10y"
        assert "full" not in {result["baseline"]["window"], result["candidate"]["window"]}


class TestDifferentStateCounts:
    def test_states_are_collapsed_when_the_counts_differ(self):
        a = fit("full", current=0, states=[state(0, 0.10, 0.05), state(1, 0.25, -0.1),
                                           state(2, 0.40, -0.3)])
        b = fit("8y", current=0, states=CALM_THEN_WILD)
        result = compare_fits(a, b)
        assert result["n_states_differ"] is True
        assert result["comparison_mode"] == "collapsed_2group"

    def test_equal_counts_compare_directly(self):
        result = compare_fits(fit("full", current=0, states=CALM_THEN_WILD),
                              fit("8y", current=1, states=CALM_THEN_WILD))
        assert result["comparison_mode"] == "direct"
        assert result["n_states_differ"] is False


class TestMissingSides:
    @pytest.mark.parametrize("baseline,candidate", [(None, "b"), ("a", None), (None, None)])
    def test_an_absent_side_is_reported_not_guessed(self, baseline, candidate):
        left = fit("full", current=0, states=CALM_THEN_WILD) if baseline else None
        right = fit("8y", current=0, states=CALM_THEN_WILD) if candidate else None
        result = compare_fits(left, right)
        assert result["status"] == "missing"
        assert result["baseline_available"] is (left is not None)
        assert result["candidate_available"] is (right is not None)


class TestTiers:
    def test_a_lone_state_is_single_not_calm(self):
        assert volatility_tiers([0.2]) == ["single"]

    def test_three_states_rank_calm_mid_high(self):
        assert volatility_tiers([0.35, 0.10, 0.22]) == ["high", "calm", "mid"]

    def test_an_unreported_current_state_is_unknown(self):
        assert tier_of(["calm", "high"], None) == "unknown"

    def test_an_out_of_range_state_is_unknown(self):
        """Better than an IndexError deep inside a report render."""
        assert tier_of(["calm", "high"], 7) == "unknown"

    def test_collapsing_anchors_calm_on_the_quietest_state(self):
        groups = calm_or_highvol([state(0, 0.40, -0.3), state(1, 0.10, -0.1)])
        assert groups[1] == "calm"

    def test_a_negative_return_state_becomes_highvol(self):
        groups = calm_or_highvol([state(0, 0.10, 0.05), state(1, 0.35, -0.2)])
        assert groups[0] == "calm"
        assert groups[1] == "highvol"


class TestSummary:
    def test_counts_every_outcome(self):
        results = [
            {"status": "ok", "agreement": "agree"},
            {"status": "ok", "agreement": "disagree"},
            {"status": "ok", "agreement": "single_state"},
            {"status": "missing"},
        ]
        assert summarise(results) == {"compared": 4, "agree": 1, "disagree": 1,
                                      "single_state": 1, "missing": 1}
