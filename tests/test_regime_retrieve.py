"""Reading a stored fit back as something the comparison can accept.

Everything here runs against a depot in ``tmp_path``. Nothing in this file may
touch a configured one.

The case worth naming: a fit is only comparable when *both* halves of it are
present — the diagnostics that describe its states, and a series point saying
which state it ended in. Half a fit must read as absent, not as a fit with a
hole in it, because a hole would be compared and would produce a verdict.
"""
from __future__ import annotations

import math

import pytest

from lazystats.io.depot import ResultDepot
from lazystats.regimes.estimation import PRODUCED_BY, PROVENANCE_SOURCE
from lazystats.regimes.persist import write_failure, write_fit
from lazystats.regimes.retrieve import load_window_fit, states_of

KEY = "regime:GLD"
DATES = ["2026-08-03", "2026-08-04", "2026-08-05"]

STATES = [
    {"state": 0, "label": "calm", "annualized_mean_return": 0.05,
     "annualized_volatility": 0.10},
    {"state": 1, "label": "wild", "annualized_mean_return": -0.20,
     "annualized_volatility": 0.35},
]


@pytest.fixture
def depot(tmp_path):
    return ResultDepot(str(tmp_path / "depot.sqlite"))


def diagnostics(**overrides) -> dict:
    base = {
        "n_states": 2, "criterion": "bic", "bic": 100.0, "loglik": -50.0,
        "n_obs": len(DATES), "labels": ["calm", "wild"], "data_start": DATES[0],
        "data_end": DATES[-1], "periods_per_year": 252, "states": STATES,
    }
    base.update(overrides)
    return base


def store_fit(depot: ResultDepot, *, key: str = KEY, current: int = 1, **overrides) -> None:
    """One fit in the depot, ending in ``current``."""
    readings = [{"state": 0, "n_states": 2, "is_high_vol": False}] * (len(DATES) - 1)
    readings.append({"state": current, "n_states": 2, "is_high_vol": current == 1})
    write_fit(depot, symbol="GLD", series_key=key, estimation_date="2026-08-05",
              diagnostics=diagnostics(**overrides), dates=DATES,
              readings=readings, retro_days=0)


class TestAStoredFitComesBackWhole:
    def test_it_carries_the_states_the_comparison_ranks_on(self, depot):
        store_fit(depot)
        fit = load_window_fit(depot, series_key=KEY, window="full")
        assert fit is not None
        assert [s["annualized_volatility"] for s in fit.states] == [0.10, 0.35]

    def test_it_reports_the_state_the_series_ended_in(self, depot):
        store_fit(depot, current=1)
        fit = load_window_fit(depot, series_key=KEY, window="full")
        assert fit is not None
        assert fit.current_state == 1
        assert fit.as_of == DATES[-1]

    def test_the_window_is_the_name_the_caller_gave_it(self, depot):
        """The key encodes a variant tag; the window's name is a different thing,
        and only the configuration knows it."""
        store_fit(depot, key="regime:GLD:8y")
        fit = load_window_fit(depot, series_key="regime:GLD:8y", window="eight_years")
        assert fit is not None
        assert fit.window == "eight_years"

    def test_it_carries_where_the_data_started(self, depot):
        store_fit(depot)
        fit = load_window_fit(depot, series_key=KEY, window="full")
        assert fit is not None
        assert fit.data_start == DATES[0]


class TestHalfAFitIsNoFit:
    def test_a_series_never_fitted_is_absent(self, depot):
        assert load_window_fit(depot, series_key="regime:NOPE", window="full") is None

    def test_a_failed_fit_is_absent_rather_than_empty(self, depot):
        write_failure(depot, symbol="GLD", series_key=KEY,
                      estimation_date="2026-08-05", error="hub unreachable")
        assert load_window_fit(depot, series_key=KEY, window="full") is None

    def test_a_row_marked_failed_is_refused_even_when_it_carries_statistics(self, depot):
        """The test above passes without the status ever being read: a failure
        row written today carries no statistics, so it reads as absent for that
        reason alone. This one puts the status and the statistics in the same
        row, which is the only shape that tells the two rules apart — and the
        shape a writer that preserved the last good diagnostics would produce."""
        depot.save(kind="regime", produced_by=PRODUCED_BY, instruments=["GLD"],
                   payload={**diagnostics(), "status": "error",
                            "error_msg": "hub unreachable",
                            "estimation_date": "2026-08-05"},
                   provenance={"source": PROVENANCE_SOURCE},
                   cadence="stable", series_key=KEY)
        depot.save_stable_point(series_key=KEY, as_of_date=DATES[-1],
                                estimation_date="2026-08-05",
                                value={"state": 1, "n_states": 2, "is_high_vol": True})
        assert load_window_fit(depot, series_key=KEY, window="full") is None

    def test_diagnostics_without_state_statistics_are_absent(self, depot):
        """A fit whose states cannot be ranked cannot be compared. Reporting it
        as a zero-state model would let it through into a verdict."""
        store_fit(depot, states=[], labels=[])
        assert load_window_fit(depot, series_key=KEY, window="full") is None

    def test_diagnostics_without_a_series_point_are_absent(self, depot):
        """Diagnostics saved, no reading appended: nothing says which state the
        symbol is in now, which is the only thing the comparison compares."""
        depot.save(kind="regime", produced_by=PRODUCED_BY, instruments=["GLD"],
                   payload={**diagnostics(), "status": "ok",
                            "estimation_date": "2026-08-05"},
                   provenance={"source": PROVENANCE_SOURCE},
                   cadence="stable", series_key=KEY)
        assert load_window_fit(depot, series_key=KEY, window="full") is None


class TestTheDepotHoldsTwoVintages:
    """Rows written before per-state statistics were persisted carry the
    engine's raw parameters instead. They are the majority of the depot's
    history and cannot be rewritten, so they must still read back."""

    def test_an_older_row_is_annualized_on_the_way_out(self, depot):
        store_fit(depot, current=0, states=None, means=[[0.0004]],
                  covars=[[[0.0001]]], labels=["calm"], n_states=1,
                  periods_per_year=252)
        fit = load_window_fit(depot, series_key=KEY, window="full")
        assert fit is not None
        assert fit.states[0]["annualized_volatility"] == pytest.approx(
            math.sqrt(0.0001 * 252))
        assert fit.states[0]["annualized_mean_return"] == pytest.approx(0.0004 * 252)

    def test_it_is_annualized_under_the_rate_its_own_row_recorded(self):
        """Not under today's constant: a weekly-fitted row annualized at 252
        would report a volatility nearly two and a half times its own."""
        weekly = states_of({"means": [[0.001]], "covars": [[[0.0004]]],
                            "labels": ["calm"], "periods_per_year": 52})
        assert weekly[0]["annualized_volatility"] == pytest.approx(
            math.sqrt(0.0004 * 52))

    def test_the_stored_statistics_win_when_both_shapes_are_present(self):
        both = states_of({"states": STATES, "means": [[9.9]], "covars": [[[9.9]]],
                          "labels": ["calm"], "periods_per_year": 252})
        assert both == STATES

    def test_neither_shape_is_no_states(self):
        assert states_of({"n_states": 2, "labels": ["calm", "wild"]}) == []
