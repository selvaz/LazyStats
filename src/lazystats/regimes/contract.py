# -*- coding: utf-8 -*-
"""
lazystats.regimes.contract — emit regime results in the shared lazydatacore envelope
==========================================================================
Converts a ``fit_regimes`` output into market-data-hub's lazydatacore
``AnalysisResult`` objects: one per series, identified by a canonical
:class:`InstrumentId` (the series name, e.g. ``ticker:SPY``), carrying the
*current* regime as a JSON-serialisable signal payload plus ``Provenance``.
This is the shared envelope every ecosystem tool stores and compares results
through — the consumer side of the same contract ``load_from_datahub`` reads.

``market-data-hub`` (which hosts lazydatacore) is an *optional* dependency,
imported lazily; install it with the ``datahub`` extra::

    pip install 'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'

Numeric policy follows the contract: regime signals are ``float`` (probabilities,
bic, loglik); the only ``Decimal`` in lazydatacore is ``Money``, which a regime
signal never produces.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Module-level handle, resolved lazily on first use and cached. Exposing it as a
# module attribute lets tests patch the dependency via
# ``monkeypatch.setattr(lazystats.regimes.contract, "lazydatacore", ...)`` — i.e. patch
# where the name is looked up — mirroring datasources.datahub.extract_returns.
lazydatacore = None  # type: ignore[assignment]


def _get_lazydatacore():
    """Return the ``market_data_hub.lazydatacore`` module (lazy import)."""
    if lazydatacore is not None:
        return lazydatacore
    try:
        import market_data_hub.lazydatacore as _dc  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "market-data-hub (lazydatacore) is required to emit AnalysisResult "
            "envelopes. Install it with the datahub extra: "
"It is a private, git-installed package: "
            "pip install 'market-data-hub @ "
            "git+https://github.com/selvaz/market-data-hub.git'."
        ) from exc
    return _dc


def _as_instrument(dc: Any, name: str) -> Any:
    """Canonical ``InstrumentId`` for a series name.

    Already-namespaced names (``ticker:SPY``, ``cik:..``) are parsed; a bare
    symbol (``SPY``) is treated as a ticker — the canonical domain for anything
    priced in ``prices_daily``.
    """
    if ":" in name:
        return dc.InstrumentId.parse(name)
    return dc.InstrumentId(domain=dc.Domain.TICKER, key=name)


def to_analysis_results(
    fit_output: Dict[str, Any],
    *,
    produced_by: str = "lazyhmm.regime.v1",
    source: str = "lazyhmm",
    tool_version: Optional[str] = None,
    as_of: Any = None,
) -> List[Any]:
    """Convert a ``fit_regimes`` output into one ``AnalysisResult`` per series.

    Accepts either shape ``fit_regimes`` produces: its compact public return
    (per series ``current_state`` / ``current_label`` / ``prob_high_vol_now``) or
    the full result stored under ``result_key`` (the ``states`` / ``prob_high_vol``
    Viterbi paths). The current-regime fields are read from whichever is present::

        {"model": str, "criterion": str, "n_timesteps": int,
         "series": {name: {"S", "labels", "bic", "loglik",
                           # compact:  "current_state", "current_label", "prob_high_vol_now"
                           # or full:   "states", "prob_high_vol", ...}}}

    Returns a list of lazydatacore ``AnalysisResult`` (``kind=SIGNAL``), each
    identified by the series' canonical ``InstrumentId`` and stamped with one
    shared ``Provenance`` (``source`` + ``as_of`` + ``tool_version``). The
    payload is the *current* regime signal — the latest state, its label, and
    the high-vol posterior — all ``float``/``int``/``str`` so the envelope
    round-trips cleanly through JSON and ``lazybridge.Store``.
    """
    dc = _get_lazydatacore()
    series: Dict[str, Any] = fit_output.get("series") or {}
    model = fit_output.get("model")
    n_timesteps = fit_output.get("n_timesteps")

    provenance = dc.Provenance(
        source=dc.SourceRef(source=source),
        as_of=as_of or dc.now_utc(),
        tool_version=tool_version,
    )

    results: List[Any] = []
    for name, sd in series.items():
        S = sd.get("S")
        labels = sd.get("labels") or []
        # Current state: the compact fit_regimes() return precomputes it; the full
        # stored result carries the Viterbi path instead. Accept either.
        states = sd.get("states") or []
        current_state = sd.get("current_state")
        if current_state is None and states:
            current_state = int(states[-1])
        current_label = sd.get("current_label")
        if (
            current_label is None
            and current_state is not None
            and current_state < len(labels)
        ):
            current_label = labels[current_state]
        # High-vol posterior "now": compact 'prob_high_vol_now', else last of the
        # full 'prob_high_vol' path.
        prob_seq = sd.get("prob_high_vol") or []
        prob_now = sd.get("prob_high_vol_now")
        if prob_now is None and prob_seq:
            prob_now = prob_seq[-1]
        # The highest-vol state is S-1 (states are ordered vol-ascending), so the
        # current bar is "high vol" iff it sits in that state — works for both shapes.
        high_vol = (
            bool(current_state == S - 1)
            if current_state is not None and S is not None
            else None
        )
        payload: Dict[str, Any] = {
            "model": model,
            "S": S,
            "current_state": current_state,
            "current_label": current_label,
            "prob_high_vol": float(prob_now) if prob_now is not None else None,
            "high_vol": high_vol,
            "n_timesteps": n_timesteps,
            "bic": sd.get("bic"),
            "loglik": sd.get("loglik"),
        }
        results.append(
            dc.AnalysisResult(
                kind=dc.ResultKind.SIGNAL,
                produced_by=produced_by,
                instruments=[_as_instrument(dc, name)],
                payload=payload,
                provenance=provenance,
            )
        )
    return results
