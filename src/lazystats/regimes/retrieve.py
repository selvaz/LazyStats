"""Reading a stored fit back out of the depot, as something comparable.

The mirror of :mod:`~lazystats.regimes.persist`. That module turns one fit into
depot rows; this one turns depot rows back into a
:class:`~lazystats.regimes.window_comparison.WindowFit` — the only shape the
comparison accepts. Nothing here refits, opens a market database, or looks at a
price: a comparison is a pure read, and keeping it that way is what lets it run
on a machine that has no access to the prices at all.

Two readings make up one fit, and they come from different places. The
*diagnostics* row says how many states the model found and what each state looks
like; the *series* says which state the symbol was in on the last stored date.
A fit missing either is not half-comparable, it is absent — so this module
returns ``None`` rather than a fit with a hole in it.

**The depot holds two vintages of the same payload.** Fits written before
per-state statistics were persisted carry the engine's raw ``means`` and
``covars``; fits written since carry the annualized ``states`` those imply.
Both are read here, and the older shape is annualized on the way out using the
``periods_per_year`` recorded beside it. This is not a compatibility shim over
an API — it is one reader over a store whose rows were genuinely written at
different times, and which cannot be rewritten: the alternative is that every
symbol reports "missing" until both of its windows have been refit.
"""
from __future__ import annotations

from typing import Any

from lazystats.io.depot import ResultDepot
from lazystats.regimes.estimation import PERIODS_PER_YEAR, annualized_states
from lazystats.regimes.window_comparison import WindowFit

#: How far back the scan for a series' newest diagnostics row looks. The depot
#: orders ``list`` by recency and offers no index from series key to result, so
#: the bound is what keeps this from reading the whole store — and, like
#: :data:`~lazystats.regimes.persist.SCAN_LIMIT`, it is a stated parameter
#: because a universe large enough to exceed it would silently report the fit
#: as absent rather than fail.
SCAN_LIMIT = 5000


def latest_regime_row(
    depot: ResultDepot, series_key: str, *, scan_limit: int = SCAN_LIMIT,
) -> dict[str, Any] | None:
    """The newest ``regime`` diagnostics row for one series, if any.

    ``depot.list`` is ordered newest first, so the first matching entry is the
    answer; a row whose fit errored is not a reading and does not count.
    """
    for entry in depot.list(cadence="stable", limit=scan_limit):
        if entry["series_key"] != series_key or entry["kind"] != "regime":
            continue
        row = depot.load(entry["result_id"])
        if row is None:
            continue
        if row["payload"].get("status") == "error":
            return None
        return row
    return None


def states_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-state annualized statistics a stored diagnostics payload implies.

    Prefers the stored ``states`` block; falls back to annualising the raw
    ``means``/``covars`` a row written before that block would carry, under the
    ``periods_per_year`` recorded with it.

    Returns:
        One record per state, or an empty list when the payload carries neither
        shape — a fit whose states cannot be ranked, which the caller must treat
        as no reading at all rather than as a zero-state model.
    """
    stored = payload.get("states")
    if isinstance(stored, list) and stored:
        return stored

    means, covars = payload.get("means"), payload.get("covars")
    if not isinstance(means, list) or not isinstance(covars, list) or not means:
        return []
    return annualized_states(
        means, covars, payload.get("labels") or [],
        periods_per_year=int(payload.get("periods_per_year") or PERIODS_PER_YEAR),
    )


def load_window_fit(
    depot: ResultDepot,
    *,
    series_key: str,
    window: str,
    scan_limit: int = SCAN_LIMIT,
) -> WindowFit | None:
    """One window's stored reading of one symbol, ready to compare.

    Args:
        depot: The depot to read.
        series_key: From :func:`~lazystats.regimes.series.series_key`, already
            carrying this window's variant tag.
        window: The window's declared name, which the comparison reports. It is
            passed in rather than parsed back out of the key: the key encodes a
            variant tag, and the name a project gave the window is a different
            thing that only the configuration knows.
        scan_limit: Bound on the search for the diagnostics row.

    Returns:
        The fit, or ``None`` when this window has no usable reading: never
        fitted, last fit errored, no state statistics, or no stored series
        point to say which state it ended in.
    """
    row = latest_regime_row(depot, series_key, scan_limit=scan_limit)
    if row is None:
        return None

    payload = row["payload"]
    states = states_of(payload)
    if not states:
        return None

    points = depot.get_series_latest(series_key)
    if not points:
        return None
    latest = points[-1]
    value = latest.get("value")
    if not isinstance(value, dict) or value.get("state") is None:
        return None

    n_states = payload.get("n_states")
    return WindowFit(
        window=window,
        # The diagnostics state the count; the statistics are what gets ranked.
        # They disagree only if a row was written inconsistently, and then the
        # statistics are the ones that decide the tiers, so they decide here too.
        n_states=int(n_states) if n_states is not None else len(states),
        current_state=int(value["state"]),
        states=tuple(states),
        as_of=latest.get("as_of_date"),
        data_start=payload.get("data_start"),
    )


__all__ = ["SCAN_LIMIT", "latest_regime_row", "load_window_fit", "states_of"]
