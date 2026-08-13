"""Contrast the regimes two estimation windows produce for the same symbol.

The question: does restricting the history change how many regimes a symbol
shows, or which one it is in now? Both fits already exist in the depot — this
module only reads them back and compares, it never refits.

**Neither window is privileged.** The previous implementation compared the whole
available history against one shorter window, with that window's tag written at
the call site. That made "eight years" part of the method, when it is a project's
choice. Here the two sides are a ``baseline`` and a ``candidate``, both named by
the caller, and either may be the unrestricted fit or a bounded one. Comparing
three years against ten is the same operation as comparing everything against
eight.

The comparison itself is symmetric: nothing below asks which side is longer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lazystats.regimes.tiers import calm_or_highvol, tier_of, volatility_tiers

#: The depot ``kind`` a saved comparison is stored under, and the series it
#: belongs to. Both are contracts with rows already in the depot: a reader
#: selecting on them finds nothing if either changes.
COMPARISON_KIND = "regime_window_comparison"
COMPARISON_SERIES_KEY = "regime_window_comparison"

#: What the comparison did, in words, recorded beside every result. A verdict
#: whose rule has to be reconstructed from the code that produced it is a
#: verdict nobody can check a year later.
CLASSIFICATION_RULE = (
    "When both windows found the same number of states, the states line up index "
    "by index — both fits order them by ascending volatility — so they are ranked "
    "directly into calm/mid/high and compared on that ranking. Collapsing here "
    "would hide a genuine mid-volatility regime both windows agree exists. Only "
    "when the state counts differ, and the states therefore cannot be lined up, "
    "is each side collapsed to two groups: the lowest-volatility state anchors "
    "'calm', and every other state joins 'calm' if its own annualized mean return "
    "is non-negative, else 'highvol'. A single-state model has nothing to rank "
    "against and is reported as 'single' rather than compared."
)


@dataclass(frozen=True)
class WindowFit:
    """One window's fitted reading of one symbol, as stored in the depot.

    Attributes:
        window: The window's name, as the configuration declares it.
        n_states: How many regimes the fit found.
        current_state: The state the symbol is in at the as-of date, or ``None``
            when the fit did not report one.
        states: Per-state statistics; each carries ``state``,
            ``annualized_volatility`` and ``annualized_mean_return``.
        as_of: The trading date this reading describes.
        data_start: The first date the fit saw.
    """

    window: str
    n_states: int
    current_state: int | None
    states: tuple[dict[str, Any], ...]
    as_of: str | None = None
    data_start: str | None = None


def compare_fits(baseline: WindowFit | None, candidate: WindowFit | None) -> dict[str, Any]:
    """Compare one symbol's readings from two windows.

    Args:
        baseline: The reading to compare against, or ``None`` if absent.
        candidate: The other reading, or ``None`` if absent.

    Returns:
        A record whose ``status`` is ``"missing"`` when either side is absent,
        otherwise ``"ok"`` with an ``agreement`` of ``"agree"``, ``"disagree"``
        or ``"single_state"``.

        When both fits found the same number of states they are ranked directly
        into calm/mid/high, because state ``i`` means the same thing in both.
        When the counts differ the states cannot be lined up, so both sides are
        collapsed to two groups first and ``comparison_mode`` says so.
    """
    if baseline is None or candidate is None:
        return {
            "status": "missing",
            "baseline_available": baseline is not None,
            "candidate_available": candidate is not None,
        }

    same_count = baseline.n_states == candidate.n_states

    if same_count:
        baseline_tier = tier_of(
            volatility_tiers([s["annualized_volatility"] for s in baseline.states]),
            baseline.current_state,
        )
        candidate_tier = tier_of(
            volatility_tiers([s["annualized_volatility"] for s in candidate.states]),
            candidate.current_state,
        )
        mode = "direct"
    else:
        baseline_groups = calm_or_highvol(list(baseline.states))
        candidate_groups = calm_or_highvol(list(candidate.states))
        # A fit that reported no current state has no group to look up, and
        # "single" is what the collapsed vocabulary calls "nothing to compare".
        baseline_tier = ("single" if baseline.current_state is None
                         else baseline_groups.get(baseline.current_state, "single"))
        candidate_tier = ("single" if candidate.current_state is None
                          else candidate_groups.get(candidate.current_state, "single"))
        mode = "collapsed_2group"

    if "single" in (baseline_tier, candidate_tier):
        agreement = "single_state"
    else:
        agreement = "agree" if baseline_tier == candidate_tier else "disagree"

    return {
        "status": "ok",
        "comparison_mode": mode,
        "agreement": agreement,
        "n_states_differ": not same_count,
        "baseline": {
            "window": baseline.window,
            "n_states": baseline.n_states,
            "current_tier": baseline_tier,
            "as_of": baseline.as_of,
            "data_start": baseline.data_start,
        },
        "candidate": {
            "window": candidate.window,
            "n_states": candidate.n_states,
            "current_tier": candidate_tier,
            "as_of": candidate.as_of,
            "data_start": candidate.data_start,
        },
    }


def summarise(comparisons: list[dict[str, Any]]) -> dict[str, int]:
    """Count the outcomes across symbols, for the report's headline."""
    return {
        "compared": len(comparisons),
        "agree": sum(1 for c in comparisons if c.get("agreement") == "agree"),
        "disagree": sum(1 for c in comparisons if c.get("agreement") == "disagree"),
        "single_state": sum(1 for c in comparisons if c.get("agreement") == "single_state"),
        "missing": sum(1 for c in comparisons if c["status"] == "missing"),
    }


def build_payload(
    readings: list[tuple[str, WindowFit | None, WindowFit | None]],
    *,
    comparison: str,
    baseline_window: str,
    candidate_window: str,
    as_of: str,
    periods_per_year: int,
    source: str,
) -> dict[str, Any]:
    """Assemble one comparison run into the record that gets stored and rendered.

    Pure: the fits arrive already loaded, so the payload's shape can be checked
    without a depot, and the stored row can be re-rendered from its own JSON
    forever after.

    Args:
        readings: One ``(symbol, baseline, candidate)`` triple per symbol. Either
            fit may be ``None``, which is what "missing" means. A symbol that
            could not be compared stays in the record, with the reason: dropping
            it would make it indistinguishable from one nobody asked about.
        comparison: The configured comparison's name.
        baseline_window: The name of the window on the baseline side.
        candidate_window: The name of the window on the candidate side.
        as_of: The date this run describes.
        periods_per_year: The rate this run annualises under. The numbers
            themselves come from each stored fit, under the rate its own row
            recorded; this states the run's.
        source: The module that produced this, for provenance.

    Returns:
        The comparison record: a per-symbol list, the counts, and the provenance
        needed to read the verdicts a year from now.
    """
    symbols: list[dict[str, Any]] = [
        {"symbol": symbol, "comparison": compare_fits(baseline, candidate)}
        for symbol, baseline, candidate in readings
    ]
    return {
        "as_of": as_of,
        "comparison": comparison,
        "baseline_window": baseline_window,
        "candidate_window": candidate_window,
        "symbols": symbols,
        "summary": summarise([s["comparison"] for s in symbols]),
        "provenance": {
            "source": source,
            "as_of": as_of,
            "comparison": comparison,
            "baseline_window": baseline_window,
            "candidate_window": candidate_window,
            "periods_per_year": periods_per_year,
            "classification_rule": CLASSIFICATION_RULE,
        },
    }


__all__ = [
    "CLASSIFICATION_RULE",
    "COMPARISON_KIND",
    "COMPARISON_SERIES_KEY",
    "WindowFit",
    "build_payload",
    "compare_fits",
    "summarise",
]
