"""Writing fits to a real depot, end to end.

Everything here runs against a depot in ``tmp_path``. Nothing in this file may
touch a configured one: these tests write, and the production depot holds
800,338 points that no test is allowed near.
"""
from __future__ import annotations

import pytest

from lazystats.io.depot import ResultDepot
from lazystats.regimes.persist import (
    find_todays_result,
    last_stored,
    write_failure,
    write_fit,
)
from lazystats.regimes.series import series_key

KEY = "regime:GLD"
DATES = [f"2026-08-{day:02d}" for day in range(1, 6)]


def reading(state: int, n_states: int = 2) -> dict:
    return {"state": state, "n_states": n_states, "is_high_vol": state == 1}


def diagnostics(n_states: int = 2, bic: float = 100.0) -> dict:
    return {"n_states": n_states, "criterion": "bic", "bic": bic,
            "loglik": -50.0, "n_obs": len(DATES), "labels": ["calm", "wild"]}


@pytest.fixture
def depot(tmp_path):
    return ResultDepot(str(tmp_path / "depot.sqlite"))


class TestFirstWrite:
    def test_a_fit_writes_diagnostics_and_every_reading(self, depot):
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                        diagnostics=diagnostics(), dates=DATES,
                        readings=[reading(0)] * 5, retro_days=2)
        assert out.result_id
        assert out.selection_reason == "first_run"
        # Append-on-change compares each trading date against *its own* prior
        # vintage, not against the previous date's reading. Nothing is stored
        # yet, so every date is new and every date is written — identical
        # readings on consecutive days are still five distinct series points.
        assert out.points_considered == 5
        assert out.points_written == 5

    def test_repeating_a_date_with_the_same_reading_adds_nothing(self, depot):
        """Where append-on-change actually bites: the same date, read again."""
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-06",
                        diagnostics=diagnostics(), dates=DATES,
                        readings=[reading(0)] * 5, retro_days=len(DATES))
        assert out.points_considered == 5
        assert out.points_written == 0

    def test_a_revised_reading_for_an_old_date_is_appended(self, depot):
        """The revision this whole design exists to capture: a past date's
        regime label changes once new data arrives."""
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-06",
                        diagnostics=diagnostics(), dates=DATES,
                        readings=[reading(0)] * 4 + [reading(1)], retro_days=len(DATES))
        assert out.points_written == 1
        vintages = depot.list_series_vintages(KEY, DATES[-1])
        assert len(vintages) == 2, "the earlier reading must survive as its own vintage"
        assert vintages[0]["value"]["state"] == 0
        assert vintages[1]["value"]["state"] == 1

    def test_the_diagnostics_carry_the_producer_the_depot_expects(self, depot):
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                        diagnostics=diagnostics(), dates=DATES,
                        readings=[reading(0)] * 5, retro_days=0)
        stored = depot.load(out.result_id)
        assert stored["produced_by"] == "scheduled:run_regime_daily"
        assert stored["payload"]["status"] == "ok"
        assert stored["payload"]["estimation_date"] == "2026-08-05"

    def test_mismatched_dates_and_readings_are_refused(self, depot):
        """Each date would be paired with another date's regime."""
        with pytest.raises(ValueError, match="paired with another"):
            write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                      diagnostics=diagnostics(), dates=DATES,
                      readings=[reading(0)] * 4, retro_days=0)


class TestSameDayRerun:
    def test_a_second_run_updates_the_diagnostics_in_place(self, depot):
        first = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                          diagnostics=diagnostics(bic=100.0), dates=DATES,
                          readings=[reading(0)] * 5, retro_days=0)
        second = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                           diagnostics=diagnostics(bic=90.0), dates=DATES,
                           readings=[reading(0)] * 5, retro_days=0)
        assert second.result_id == first.result_id
        assert depot.load(first.result_id)["payload"]["bic"] == 90.0

    def test_a_failure_does_not_erase_the_days_successful_fit(self, depot):
        """The rule, exercised against a real depot rather than in the
        abstract: the morning's parameters must still be readable."""
        ok = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                       diagnostics=diagnostics(bic=100.0), dates=DATES,
                       readings=[reading(0)] * 5, retro_days=0)
        failed = write_failure(depot, symbol="GLD", series_key=KEY,
                               estimation_date="2026-08-05", error="network down")
        assert failed.result_id is None
        stored = depot.load(ok.result_id)
        assert stored["payload"]["status"] == "ok"
        assert stored["payload"]["bic"] == 100.0

    def test_a_failure_replaces_an_earlier_failure(self, depot):
        first = write_failure(depot, symbol="GLD", series_key=KEY,
                              estimation_date="2026-08-05", error="first")
        second = write_failure(depot, symbol="GLD", series_key=KEY,
                               estimation_date="2026-08-05", error="second")
        assert second.result_id == first.result_id
        assert depot.load(first.result_id)["payload"]["error_msg"] == "second"

    def test_todays_result_is_found_by_series_and_date(self, depot):
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                        diagnostics=diagnostics(), dates=DATES,
                        readings=[reading(0)] * 5, retro_days=0)
        assert find_todays_result(depot, KEY, "2026-08-05") == (out.result_id, "ok")
        assert find_todays_result(depot, KEY, "2026-08-04") == (None, None)

    def test_another_series_same_day_is_not_confused_for_this_one(self, depot):
        write_fit(depot, symbol="SPY", series_key="regime:SPY",
                  estimation_date="2026-08-05", diagnostics=diagnostics(),
                  dates=DATES, readings=[reading(0)] * 5, retro_days=0)
        assert find_todays_result(depot, KEY, "2026-08-05") == (None, None)


class TestIncrementalDays:
    def test_the_next_day_writes_only_what_is_new(self, depot):
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        assert last_stored(depot, KEY) == ("2026-08-05", 2)

        extended = DATES + ["2026-08-06"]
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-06",
                        diagnostics=diagnostics(), dates=extended,
                        readings=[reading(0)] * 5 + [reading(1)], retro_days=0)
        assert out.selection_reason == "incremental"
        assert out.points_considered == 1
        assert out.points_written == 1

    def test_a_state_count_change_rewrites_the_whole_history(self, depot):
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(n_states=2), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-06",
                        diagnostics=diagnostics(n_states=3), dates=DATES,
                        readings=[reading(0, n_states=3)] * 5, retro_days=0)
        assert out.selection_reason == "model_change"
        assert out.points_considered == len(DATES)


class TestVariantsStaySeparate:
    def test_a_windowed_fit_does_not_touch_the_full_history_series(self, depot):
        full = series_key("GLD", market_db="/db", production_db="/db")
        windowed = series_key("GLD", market_db="/db", production_db="/db", variant="8y")

        write_fit(depot, symbol="GLD", series_key=full, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        write_fit(depot, symbol="GLD", series_key=windowed, estimation_date="2026-08-05",
                  diagnostics=diagnostics(n_states=3), dates=DATES,
                  readings=[reading(2, n_states=3)] * 5, retro_days=0)

        assert last_stored(depot, full)[1] == 2
        assert last_stored(depot, windowed)[1] == 3

    def test_a_canonical_id_would_have_started_a_parallel_series(self, depot):
        """Stated as the consequence: this is what the boundary prevents."""
        good = series_key("ticker:GLD", market_db="/db", production_db="/db")
        assert good == "regime:GLD"
        write_fit(depot, symbol="GLD", series_key=good, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 5, retro_days=0)
        assert last_stored(depot, "regime:ticker:GLD") == (None, None)
        assert last_stored(depot, "regime:GLD")[0] == "2026-08-05"


class TestWhatCountsAsAChange:
    """A reading carries both the discrete regime call and a probability. Only
    the first decides whether the reading changed: the second moves on every
    refit merely from one more day of data."""

    def test_a_moved_probability_alone_is_not_a_revision(self):
        """The trap this guards: comparing the whole reading would report a
        retroactive revision for every date in the window, on every run, for
        every symbol — burying the real ones."""
        depot = ResultDepot(":memory:")
        base = [{"state": 0, "n_states": 2, "is_high_vol": False,
                 "prob_high_vol": 0.11}] * len(DATES)
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES, readings=base, retro_days=0)

        drifted = [{"state": 0, "n_states": 2, "is_high_vol": False,
                    "prob_high_vol": 0.4287} for _ in DATES]
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-06", diagnostics=diagnostics(),
                        dates=DATES, readings=drifted, retro_days=len(DATES))
        assert out.points_written == 0
        assert out.changed_dates == ()

    def test_a_moved_state_is_a_revision_and_is_named(self):
        depot = ResultDepot(":memory:")
        calm = [{"state": 0, "n_states": 2, "is_high_vol": False,
                 "prob_high_vol": 0.11}] * len(DATES)
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES, readings=calm, retro_days=0)

        revised = list(calm)
        revised[1] = {"state": 1, "n_states": 2, "is_high_vol": True,
                      "prob_high_vol": 0.93}
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-06", diagnostics=diagnostics(),
                        dates=DATES, readings=revised, retro_days=len(DATES))
        assert out.changed_dates == (DATES[1],)
        assert out.points_written == 1

    def test_the_probability_is_stored_even_though_it_is_not_compared(self):
        """Not compared is not the same as not kept: the report prints it."""
        depot = ResultDepot(":memory:")
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-05",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[{"state": 0, "n_states": 2, "is_high_vol": False,
                             "prob_high_vol": 0.1234}] * len(DATES),
                  retro_days=0)
        assert depot.get_series_latest(KEY)[-1]["value"]["prob_high_vol"] == 0.1234

    def test_every_date_a_first_run_wrote_is_named(self):
        depot = ResultDepot(":memory:")
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-05", diagnostics=diagnostics(),
                        dates=DATES, readings=[reading(0)] * len(DATES), retro_days=0)
        assert list(out.changed_dates) == DATES


# ---------------------------------------------------------------------------
# "changed today" has to mean the regime moved, not that a point was written.
# Observed in production on 2026-08-16: the report flagged 109 of 109
# instruments as changed while the depot held five real state changes.
# ---------------------------------------------------------------------------


def _lettura(state, n_states=3, is_high_vol=False, prob=0.1):
    return {"state": state, "n_states": n_states, "is_high_vol": is_high_vol,
            "prob_high_vol": prob, "state_probs": [0.5, 0.4, 0.1]}


def test_regime_changed_compares_the_call_not_the_probabilities():
    from lazystats.regimes.persist import regime_changed

    assert regime_changed(_lettura(1), _lettura(0)) is True
    assert regime_changed(_lettura(1), _lettura(1)) is False
    # a probability moves on every refit from one more day of data; on its own
    # it is not a regime change, and treating it as one would flag everything
    assert regime_changed(_lettura(1, prob=0.10), _lettura(1, prob=0.93)) is False
    # the other two keys do count
    assert regime_changed(_lettura(1, n_states=3), _lettura(1, n_states=2)) is True
    assert regime_changed(_lettura(1, is_high_vol=False),
                          _lettura(1, is_high_vol=True)) is True


def test_a_series_with_one_reading_has_not_changed():
    """Nothing to compare against is not a change. It used to be one, because
    the newest date is always newly written."""
    from lazystats.regimes.persist import regime_changed

    assert regime_changed(None, _lettura(0)) is False


def test_a_written_point_is_not_by_itself_a_regime_change(tmp_path):
    """The defect, end to end.

    `write_fit` reports `changed_dates` -- the dates a point was written for --
    and the store writes whenever it holds nothing for that date yet. On the
    newest trading date that is true every single day, so reading
    `newest in changed_dates` as "the regime changed" flagged every instrument
    on every run.
    """
    from lazystats.io.depot import ResultDepot
    from lazystats.regimes.persist import regime_changed, write_fit

    depot = ResultDepot(str(tmp_path / "d.sqlite"))
    dates = ["2026-08-13", "2026-08-14"]
    readings = [_lettura(1), _lettura(1)]        # stesso stato: NON e' un cambio

    written = write_fit(
        depot, symbol="AAA", series_key="regime:AAA",
        estimation_date="2026-08-14",
        diagnostics={"n_states": 3}, dates=dates, readings=readings,
        retro_days=5,
    )

    # il punto piu' recente e' stato scritto -- e' un giorno nuovo
    assert dates[-1] in written.changed_dates
    # ma il regime non si e' mosso, ed e' questo che il report deve dire
    assert regime_changed(readings[-2], readings[-1]) is False
