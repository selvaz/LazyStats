"""The daily run, as one record that can be stored and re-read.

Two reports come out of the same fit. The chart-based one is assembled while the
models are alive, because its plots need them
(:mod:`~lazystats.regimes.report`). This one is different: it carries no images,
so it fits in the depot, and a saved row can be rendered again years later
without refitting anything — which is what makes yesterday's reading auditable
rather than merely remembered.

Both are built from the *same* records. Its predecessor kept two parallel
descriptions of one run — a result object for the charts, and a payload builder
that went back to the depot to re-read the fits — and the two could disagree
about what the day looked like without anything failing.

Pure: the records arrive already assembled, so the shape can be checked without
a depot.
"""
from __future__ import annotations

from typing import Any

from lazystats.regimes.report import SymbolReport
from lazystats.regimes.tiers import volatility_tiers

#: The depot ``kind`` a saved daily report is stored under, and its series. Both
#: are contracts with rows already in the depot.
REPORT_KIND = "regime_daily_report"
REPORT_SERIES_KEY = "regime_daily_report"


def _states_with_tiers(entry: SymbolReport) -> list[dict[str, Any]]:
    """Each state's statistics with its calm/mid/high tag attached.

    Ranked here rather than in the page's script. Its predecessor carried the
    same ranking twice — once in Python for the chart report, once in JavaScript
    for this one — with a comment hoping they would not drift. One of them
    decides what counts as "mid"; it is :mod:`~lazystats.regimes.tiers`.
    """
    tiers = volatility_tiers([s["annualized_volatility"] for s in entry.states])
    return [{**state, "tier": tier} for state, tier in zip(entry.states, tiers)]


def _symbol_record(entry: SymbolReport) -> dict[str, Any]:
    return {
        "symbol": entry.symbol,
        "name": entry.name,
        "n_states": entry.n_states,
        "current_state": entry.current_state,
        "current_label": entry.current_label,
        "current_tier": entry.current_tier,
        "current_state_probs": list(entry.current_state_probs),
        "is_high_vol": entry.is_high_vol,
        "changed_today": entry.changed_today,
        # The count the report filters on, and the dates behind it. The count
        # alone cannot be checked against anything.
        "revised": len(entry.revisions),
        "revised_dates": [r.trading_date for r in entry.revisions],
        "states": _states_with_tiers(entry),
        "fit": {
            "bic": entry.bic,
            "loglik": entry.loglik,
            "data_start": entry.data_start,
            "data_end": entry.data_end,
            "n_obs": entry.n_obs,
        },
    }


def build_payload(entries: list[SymbolReport], *, as_of: str, periods_per_year: int,
                  source: str) -> dict[str, Any]:
    """Assemble one daily run into the record that gets stored and rendered.

    Args:
        entries: One record per symbol, fitted or failed.
        as_of: The date the run describes.
        periods_per_year: The rate the annualized statistics were computed at.
        source: The module that produced this, for provenance.

    Returns:
        The run's record: the fitted symbols, the failed ones with their
        reasons, the counts, and the provenance.
    """
    fitted = sorted((e for e in entries if e.ok), key=lambda e: e.symbol)
    failed = sorted((e for e in entries if not e.ok), key=lambda e: e.symbol)

    return {
        "as_of": as_of,
        "symbols": [_symbol_record(e) for e in fitted],
        "errors": [{"symbol": e.symbol, "name": e.name, "error_msg": e.error}
                   for e in failed],
        "summary": {
            "n_ok": len(fitted),
            "n_errors": len(failed),
            "n_changed_today": sum(1 for e in fitted if e.changed_today),
            "n_revised": sum(1 for e in fitted if e.revisions),
        },
        "provenance": {
            "source": source,
            "as_of": as_of,
            "periods_per_year": periods_per_year,
        },
    }


__all__ = ["REPORT_KIND", "REPORT_SERIES_KEY", "build_payload"]
