"""Deciding which of a fit's readings to write back.

A run refits a symbol's whole history, but almost none of that is news: the
regime label for a trading date five years ago is nearly always the same one
already stored. Writing all of it every day would bury the interesting rows —
the revisions — under thousands of identical ones.

So a run writes a *window*, and choosing that window has three rules. They read
as arithmetic but each one exists because of a failure:

**First run.** Nothing is stored, so everything is.

**The model changed shape.** When the criterion picks a different number of
states, every state is renumbered: state 1 in a two-state fit is not state 1 in
a three-state fit. Writing only the recent window would leave the newest vintage
mixing two incompatible numberings, so the whole history is re-appended as one
consistent vintage. Older vintages stay — a past reading is never overwritten,
only superseded.

**Otherwise, the retro window *plus everything newer than what is stored*.** The
second half is what stops a gap becoming permanent: after a pause longer than
the retro window, a run that wrote only the last thirty days would skip the
dates in between, and no later run would ever revisit them.

The decision is kept apart from the writing because this is where the damage
would be, and none of it needs a database to check.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    """Which readings to write, and why.

    Attributes:
        indices: Positions into the run's dates, ascending.
        reason: ``"first_run"``, ``"model_change"`` or ``"incremental"``.
    """

    indices: tuple[int, ...]
    reason: str

    def __len__(self) -> int:
        return len(self.indices)


def select_rows(
    dates: list[str],
    *,
    last_stored_date: str | None,
    last_n_states: int | None,
    n_states: int,
    retro_days: int,
) -> Selection:
    """Choose which of this run's readings to append.

    Args:
        dates: The run's trading dates, ascending.
        last_stored_date: The most recent date already in the depot for this
            series, or ``None`` when nothing is stored.
        last_n_states: The state count of the stored reading, or ``None``.
        n_states: The state count this run found.
        retro_days: How many of the most recent readings to revisit even when
            they are already stored. Zero writes only what is new.

    Returns:
        The positions to write and the rule that chose them.

    Raises:
        ValueError: ``dates`` is not ascending, or ``retro_days`` is negative.
            Both would silently select the wrong rows.
    """
    if retro_days < 0:
        raise ValueError(f"retro_days must not be negative, got {retro_days}")
    if any(b < a for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("dates must be ascending; the selection depends on their order")
    if not dates:
        return Selection(indices=(), reason="first_run" if last_stored_date is None
                         else "incremental")

    if last_stored_date is None:
        return Selection(indices=tuple(range(len(dates))), reason="first_run")

    if last_n_states is not None and last_n_states != n_states:
        return Selection(indices=tuple(range(len(dates))), reason="model_change")

    chosen = {i for i, d in enumerate(dates) if d > last_stored_date}
    if retro_days > 0:
        chosen.update(range(max(0, len(dates) - retro_days), len(dates)))
    return Selection(indices=tuple(sorted(chosen)), reason="incremental")


__all__ = ["Selection", "select_rows"]
