"""Reading a fitted model's states as calm, mid or high volatility.

A fit gives back states numbered arbitrarily; what a reader wants to know is
which one is the quiet regime and which the turbulent one. These two functions
are the whole of that interpretation, and they live together because they must
not drift apart: one ranks states within a single fit, the other collapses two
fits with different state counts onto a common footing so they can be compared.
"""
from __future__ import annotations

from typing import Any


def volatility_tiers(annualized_vols: list[float]) -> list[str]:
    """Rank states by volatility, returning a tier per state, in order.

    The lowest is ``"calm"``, the highest ``"high"``, anything between is
    ``"mid"``. A model with one state has nothing to rank against, so its single
    state is ``"single"`` rather than being called calm by default.

    Three tiers rather than two on purpose: collapsing here would hide a genuine
    mid-volatility regime that a fit actually found.
    """
    n = len(annualized_vols)
    if n <= 1:
        return ["single"] * n
    order = sorted(range(n), key=lambda i: annualized_vols[i])
    tiers = [""] * n
    for rank, index in enumerate(order):
        tiers[index] = "calm" if rank == 0 else "high" if rank == n - 1 else "mid"
    return tiers


def calm_or_highvol(states: list[dict[str, Any]]) -> dict[int, str]:
    """Collapse a model's states to exactly two groups, keyed by state number.

    Used when two fits disagree about how many states exist: their states cannot
    be compared index by index, so both are reduced to the same two groups
    first. The lowest-volatility state anchors ``"calm"``; every state, that
    anchor included, then joins ``"calm"`` if its own annualized mean return is
    non-negative and ``"highvol"`` otherwise.

    A single-state model has nothing to rank against and reports ``"single"``.
    """
    if len(states) <= 1:
        return {st["state"]: "single" for st in states}
    lowest_vol = min(states, key=lambda st: st["annualized_volatility"])["state"]
    groups: dict[int, str] = {}
    for st in states:
        if st["state"] == lowest_vol:
            groups[st["state"]] = "calm"
        else:
            groups[st["state"]] = "calm" if st["annualized_mean_return"] >= 0 else "highvol"
    return groups


def tier_of(tiers: list[str], state: int | None) -> str:
    """The tier of one state, or ``"unknown"`` when the state is not reported.

    A fit can come back without a current state — too little data, a failed
    run — and calling that ``"calm"`` would be an assertion nobody made.
    """
    if state is None or not (0 <= state < len(tiers)):
        return "unknown"
    return tiers[state]


__all__ = ["calm_or_highvol", "tier_of", "volatility_tiers"]
