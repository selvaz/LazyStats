"""Writing one symbol's fit to the depot.

Everything decided elsewhere: :mod:`~lazystats.regimes.series` says under which
key, :mod:`~lazystats.regimes.vintage` says which readings, and
:mod:`~lazystats.regimes.rerun` says what to do about a second attempt today.
This module does the writing, and nothing else — which is why it is short, and
why the rules above can be checked without a database.

Two results come out of one fit. A ``regime`` result carries the diagnostics:
the criterion, the fitted parameters, how long it took. The readings themselves
go in as stable series points, one per trading date, appended only where the
label actually changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lazystats.io.depot import ResultDepot
from lazystats.regimes.estimation import PRODUCED_BY, PROVENANCE_SOURCE
from lazystats.regimes.rerun import decide
from lazystats.regimes.vintage import select_rows

#: How far back the scan for today's existing diagnostics result looks. The
#: depot has no query on (series_key, estimation_date), so the most recent
#: stable entries are scanned instead. Same-day reruns — the only case this
#: needs to catch — sit at the front of that list.
SCAN_LIMIT = 2000

#: The keys that decide whether a reading *changed*. Only the discrete regime
#: call: a probability shifts on every refit merely from one more day of data,
#: so comparing the whole reading would report a retroactive revision for every
#: date in the window, on every run, for every symbol. The full reading is still
#: what gets stored.
CHANGE_KEYS = ["state", "n_states", "is_high_vol"]


@dataclass(frozen=True)
class WriteOutcome:
    """What a write actually did.

    Attributes:
        result_id: The diagnostics result written, or ``None`` if refused.
        points_written: How many readings changed and were appended.
        points_considered: How many readings the rules selected.
        reason: Why the diagnostics were written, replaced or skipped.
        selection_reason: Which rule chose the readings.
        changed_dates: The trading dates whose reading actually changed, in
            order. A report distinguishes a *revision* — an old date whose
            regime call moved — from the newest date, which is simply new; that
            needs the dates themselves, not how many there were.
    """

    result_id: str | None
    points_written: int
    points_considered: int
    reason: str
    selection_reason: str
    changed_dates: tuple[str, ...] = ()


def find_todays_result(
    depot: ResultDepot, series_key: str, estimation_date: str,
    *, scan_limit: int = SCAN_LIMIT,
) -> tuple[str | None, str | None]:
    """The diagnostics result already stored for this series and date.

    Returns ``(result_id, status)``, or ``(None, None)``. The scan is bounded:
    a universe large enough to push a same-day rerun past ``scan_limit`` would
    make this silently miss it and write a duplicate, so the bound is a stated
    parameter rather than a constant buried in the loop.
    """
    for entry in depot.list(cadence="stable", limit=scan_limit):
        if entry["series_key"] != series_key:
            continue
        full = depot.load(entry["result_id"])
        if full is None:
            continue
        payload = full["payload"]
        if payload.get("estimation_date") == estimation_date:
            return entry["result_id"], payload.get("status")
    return None, None


def write_fit(
    depot: ResultDepot,
    *,
    symbol: str,
    series_key: str,
    estimation_date: str,
    diagnostics: dict[str, Any],
    dates: list[str],
    readings: list[Any],
    retro_days: int,
    scan_limit: int = SCAN_LIMIT,
) -> WriteOutcome:
    """Record one successful fit: its diagnostics, then its readings.

    Args:
        depot: Where to write.
        symbol: The bare symbol, as the key already reflects.
        series_key: From :func:`~lazystats.regimes.series.series_key`.
        estimation_date: The date this run was made.
        diagnostics: The fit's parameters and metadata. ``status`` and
            ``estimation_date`` are set here, not by the caller.
        dates: Trading dates, ascending.
        readings: One reading per date, in the same order.
        retro_days: How far back to revisit already-stored readings.
        scan_limit: Bound on the same-day lookup.

    Returns:
        What was written.

    Raises:
        ValueError: ``dates`` and ``readings`` differ in length, which would
            silently pair each date with the wrong reading.
    """
    if len(dates) != len(readings):
        raise ValueError(
            f"{len(dates)} dates and {len(readings)} readings: each date would be "
            f"paired with another date's regime"
        )

    existing_id, existing_status = find_todays_result(
        depot, series_key, estimation_date, scan_limit=scan_limit
    )
    call = decide(outcome="ok", existing_result_id=existing_id,
                  existing_outcome=existing_status)

    payload = dict(diagnostics)
    payload["status"] = "ok"
    payload["error_msg"] = None
    payload["estimation_date"] = estimation_date

    result_id = depot.save(
        kind="regime",
        produced_by=PRODUCED_BY,
        instruments=[symbol],
        payload=payload,
        provenance={"source": PROVENANCE_SOURCE},
        cadence="stable",
        series_key=series_key,
        result_id=call.replaces,
    )

    last_date, last_states = last_stored(depot, series_key)
    selection = select_rows(
        dates,
        last_stored_date=last_date,
        last_n_states=last_states,
        n_states=int(diagnostics.get("n_states", 0)),
        retro_days=retro_days,
    )

    changed: list[str] = []
    for index in selection.indices:
        if depot.save_stable_point(
            series_key=series_key,
            as_of_date=dates[index],
            estimation_date=estimation_date,
            value=readings[index],
            result_id=result_id,
            compare_keys=CHANGE_KEYS,
        ):
            changed.append(dates[index])

    return WriteOutcome(
        result_id=result_id,
        points_written=len(changed),
        points_considered=len(selection),
        reason=call.reason,
        selection_reason=selection.reason,
        changed_dates=tuple(changed),
    )


def write_failure(
    depot: ResultDepot,
    *,
    symbol: str,
    series_key: str,
    estimation_date: str,
    error: str,
    scan_limit: int = SCAN_LIMIT,
) -> WriteOutcome:
    """Record that a fit failed, unless today already has a successful one."""
    existing_id, existing_status = find_todays_result(
        depot, series_key, estimation_date, scan_limit=scan_limit
    )
    call = decide(outcome="error", existing_result_id=existing_id,
                  existing_outcome=existing_status)

    if not call.write:
        return WriteOutcome(result_id=None, points_written=0, points_considered=0,
                            reason=call.reason, selection_reason="none")

    result_id = depot.save(
        kind="regime",
        produced_by=PRODUCED_BY,
        instruments=[symbol],
        payload={
            "status": "error",
            "error_msg": error[:500],
            "estimation_date": estimation_date,
        },
        provenance={"source": PROVENANCE_SOURCE},
        cadence="stable",
        series_key=series_key,
        result_id=call.replaces,
    )
    return WriteOutcome(result_id=result_id, points_written=0, points_considered=0,
                        reason=call.reason, selection_reason="none")


def last_stored(depot: ResultDepot, series_key: str) -> tuple[str | None, int | None]:
    """The most recent stored reading's date, and the state count behind it.

    Both feed :func:`~lazystats.regimes.vintage.select_rows`: the date decides
    what is new, the state count decides whether the model changed shape.

    The depot offers no "latest point" query, so this reads the series and takes
    its last row — the rows are ordered ascending by date. That is more work
    than the answer needs on a series with a decade of history, and worth
    replacing with a targeted query if this ever runs per-symbol on a large
    universe; it is correct, which comes first.
    """
    series = depot.get_series_latest(series_key)
    if not series:
        return None, None
    latest = series[-1]
    value = latest.get("value")
    states = value.get("n_states") if isinstance(value, dict) else None
    return latest.get("as_of_date"), states


__all__ = ["CHANGE_KEYS", "SCAN_LIMIT", "WriteOutcome", "find_todays_result",
           "last_stored", "write_failure", "write_fit"]
