"""Which readings a run writes back.

Every rule here exists because of a way the history can be corrupted without
anything failing: a gap that no later run revisits, or a vintage that mixes two
incompatible state numberings. The tests are written as those failures.
"""
from __future__ import annotations

import pytest

from lazystats.regimes.vintage import select_rows

# Ten consecutive trading dates, oldest first.
DATES = [f"2026-08-{day:02d}" for day in range(1, 11)]


def selected(sel, dates=DATES):
    return [dates[i] for i in sel.indices]


class TestFirstRun:
    def test_everything_is_written_when_nothing_is_stored(self):
        sel = select_rows(DATES, last_stored_date=None, last_n_states=None,
                          n_states=2, retro_days=3)
        assert sel.reason == "first_run"
        assert len(sel) == len(DATES)

    def test_an_empty_fit_selects_nothing(self):
        sel = select_rows([], last_stored_date=None, last_n_states=None,
                          n_states=2, retro_days=3)
        assert len(sel) == 0


class TestModelChange:
    """A different state count renumbers every state."""

    def test_the_whole_history_is_rewritten(self):
        sel = select_rows(DATES, last_stored_date="2026-08-08", last_n_states=2,
                          n_states=3, retro_days=3)
        assert sel.reason == "model_change"
        assert selected(sel) == DATES

    def test_writing_only_the_recent_window_would_mix_numberings(self):
        """The point of the rule: state 1 in a two-state fit is not state 1 in a
        three-state one, so a vintage must not contain both."""
        sel = select_rows(DATES, last_stored_date="2026-08-09", last_n_states=3,
                          n_states=2, retro_days=1)
        assert len(sel) == len(DATES)

    def test_an_unchanged_count_is_not_a_model_change(self):
        sel = select_rows(DATES, last_stored_date="2026-08-08", last_n_states=2,
                          n_states=2, retro_days=3)
        assert sel.reason == "incremental"

    def test_an_unknown_previous_count_is_not_treated_as_a_change(self):
        """Absent is not different: rewriting everything on missing metadata
        would make an ordinary run look like a model flip."""
        sel = select_rows(DATES, last_stored_date="2026-08-08", last_n_states=None,
                          n_states=2, retro_days=2)
        assert sel.reason == "incremental"


class TestIncremental:
    def test_new_dates_are_always_written(self):
        sel = select_rows(DATES, last_stored_date="2026-08-08", last_n_states=2,
                          n_states=2, retro_days=0)
        assert selected(sel) == ["2026-08-09", "2026-08-10"]

    def test_the_retro_window_revisits_already_stored_dates(self):
        """A label for an old date can change when new data arrives."""
        sel = select_rows(DATES, last_stored_date="2026-08-10", last_n_states=2,
                          n_states=2, retro_days=3)
        assert selected(sel) == ["2026-08-08", "2026-08-09", "2026-08-10"]

    def test_a_pause_longer_than_the_retro_window_still_backfills(self):
        """The failure this rule exists for. With only the retro window, the
        dates between the last stored one and the window's start would be
        skipped, and no later run would ever return to them."""
        sel = select_rows(DATES, last_stored_date="2026-08-02", last_n_states=2,
                          n_states=2, retro_days=2)
        assert selected(sel) == DATES[2:], "a gap was left that nothing would backfill"

    def test_nothing_is_written_when_nothing_is_new_and_there_is_no_retro(self):
        sel = select_rows(DATES, last_stored_date="2026-08-10", last_n_states=2,
                          n_states=2, retro_days=0)
        assert len(sel) == 0

    def test_a_retro_window_larger_than_the_history_is_harmless(self):
        sel = select_rows(DATES, last_stored_date="2026-08-10", last_n_states=2,
                          n_states=2, retro_days=999)
        assert selected(sel) == DATES

    def test_the_selection_has_no_duplicates_where_the_rules_overlap(self):
        """New dates and the retro window overlap; a row written twice would
        append two readings for one date in the same vintage."""
        sel = select_rows(DATES, last_stored_date="2026-08-07", last_n_states=2,
                          n_states=2, retro_days=5)
        assert len(sel.indices) == len(set(sel.indices))
        assert list(sel.indices) == sorted(sel.indices)


class TestRefusesAmbiguousInput:
    def test_unordered_dates_are_refused(self):
        """Every rule compares dates; out of order, they select the wrong rows
        and report success."""
        with pytest.raises(ValueError, match="ascending"):
            select_rows(["2026-08-03", "2026-08-01"], last_stored_date="2026-08-01",
                        last_n_states=2, n_states=2, retro_days=1)

    def test_a_negative_retro_window_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            select_rows(DATES, last_stored_date="2026-08-05", last_n_states=2,
                        n_states=2, retro_days=-1)
