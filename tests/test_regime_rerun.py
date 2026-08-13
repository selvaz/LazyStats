"""Fitting the same symbol twice in one day.

The rule worth guarding is asymmetric: a success may replace anything, a failure
may not replace a success. Losing a day's fitted parameters to an evening
network error is silent — the readings stay, and only the diagnostics they point
at change to say the fit failed.
"""
from __future__ import annotations

import pytest

from lazystats.regimes.rerun import decide


class TestNothingStoredYet:
    @pytest.mark.parametrize("outcome", ["ok", "error"])
    def test_the_first_attempt_always_writes(self, outcome):
        d = decide(outcome=outcome, existing_result_id=None, existing_outcome=None)
        assert d.write
        assert d.replaces is None


class TestSecondAttemptUpdatesInPlace:
    """Two diagnostics results for one day would make "what did the model look
    like on the 12th?" ambiguous."""

    def test_a_success_replaces_an_earlier_success(self):
        d = decide(outcome="ok", existing_result_id="res_1", existing_outcome="ok")
        assert d.write
        assert d.replaces == "res_1"

    def test_a_success_replaces_an_earlier_failure(self):
        d = decide(outcome="ok", existing_result_id="res_1", existing_outcome="error")
        assert d.write
        assert d.replaces == "res_1"

    def test_a_failure_replaces_an_earlier_failure(self):
        d = decide(outcome="error", existing_result_id="res_1", existing_outcome="error")
        assert d.write
        assert d.replaces == "res_1"


class TestAFailureNeverReplacesASuccess:
    """The rule this module exists for."""

    def test_the_evening_failure_is_not_written(self):
        d = decide(outcome="error", existing_result_id="res_1", existing_outcome="ok")
        assert not d.write
        assert d.replaces is None

    def test_the_reason_says_why_nothing_was_written(self):
        """It has to be legible in a log: a run that wrote nothing and exited
        cleanly is otherwise indistinguishable from one that did nothing."""
        d = decide(outcome="error", existing_result_id="res_1", existing_outcome="ok")
        assert "refusing to replace" in d.reason

    def test_the_morning_parameters_survive_the_evening_error(self):
        """Stated as the scenario rather than the branch: the day's fit is
        recorded, a later attempt fails, and the fit must still be there."""
        morning = decide(outcome="ok", existing_result_id=None, existing_outcome=None)
        assert morning.write

        evening = decide(outcome="error", existing_result_id="res_morning",
                         existing_outcome="ok")
        assert not evening.write, "the day's fitted parameters would have been lost"
