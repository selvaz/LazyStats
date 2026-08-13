"""What to do when a symbol is fitted more than once on the same day.

Each run records one fit-diagnostics result per symbol per day: the criterion,
the fitted parameters, how long it took, whether it worked. Fitting the same
symbol twice for one estimation date must not leave two of those — a reader
asking "what did the model look like on the 12th?" would get an arbitrary one.

So a second attempt updates the first in place. With one exception, and it is
the reason this is a function rather than an ``if``:

**A failed rerun must not replace a successful earlier run.** The morning run
fits, records its parameters, and writes its readings. An evening rerun hits a
network error and fails. Overwriting would trade a real fit for an error
message, losing the day's parameters entirely — and the readings the morning
wrote would then point at a diagnostics result that says the fit failed.

A failure only ever replaces another failure, or writes where nothing is.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """What a run should do with its diagnostics result.

    Attributes:
        write: Whether to write at all.
        replaces: The result to update in place, or ``None`` to create one.
        reason: Why, in words a log line can carry.
    """

    write: bool
    replaces: str | None
    reason: str


def decide(
    *,
    outcome: str,
    existing_result_id: str | None,
    existing_outcome: str | None,
) -> Decision:
    """Decide how this attempt's diagnostics should be recorded.

    Args:
        outcome: ``"ok"`` or ``"error"`` for the attempt just made.
        existing_result_id: The diagnostics result already stored for this
            symbol and estimation date, if any.
        existing_outcome: That result's outcome, if any.

    Returns:
        Whether to write, and which result to replace.
    """
    if existing_result_id is None:
        return Decision(write=True, replaces=None, reason="first attempt today")

    if outcome == "error" and existing_outcome == "ok":
        return Decision(
            write=False,
            replaces=None,
            reason="a successful run already recorded today; refusing to replace it "
                   "with a failure",
        )

    if outcome == "error":
        return Decision(write=True, replaces=existing_result_id,
                        reason="replacing an earlier failure")

    return Decision(write=True, replaces=existing_result_id,
                    reason="updating today's result in place")


__all__ = ["Decision", "decide"]
