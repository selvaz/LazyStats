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
