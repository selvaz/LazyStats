"""Daily regime estimation: the full-refit pipeline and what it writes.

Every run refits each symbol over its whole available return history, rather
than applying fixed parameters, which is what makes it possible to watch whether
one more day of data changes the model's reading of the past. Results go to the
shared result depot with append-on-change semantics: a past
``(symbol, trading_date, estimation_date)`` reading is never overwritten, only
superseded once the discretized label actually differs.

This module owns the pipeline. It does not own the engine — that is
:mod:`lazystats.regimes.core` — and it does not own the prices, which arrive
through :func:`lazystats.io.datahub.load_returns` like every other consumer's.

Two things here are *contracts with data already in the depot*, not choices:

``PRODUCED_BY`` is the identity 1,809 existing results were written under, and
downstream jobs select by it. ``series.series_key`` builds the keys the 405,313
row migration used. Change either and nothing fails — new rows simply land in a
place no reader is looking, and the histories restart in silence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lazystats.io.datahub import load_returns
from lazystats.models import ReturnDataset
from lazystats.regimes.series import bare_symbol

#: Trading periods in a year, for annualising a fit made on daily returns. The
#: fit itself is frequency-agnostic; this number is what turns its per-period
#: parameters into the quantities a reader compares. It is recorded next to the
#: statistics it produced, so a row can never be read under a different one.
PERIODS_PER_YEAR = 252

#: The producer identity every regime result is written under. Downstream
#: selection depends on it; it survives the module moving between repositories
#: because it names the *job*, not the code's location.
PRODUCED_BY = "scheduled:run_regime_daily"

#: Where the code that produced a row lives. Unlike PRODUCED_BY this one is
#: allowed to change when the code moves, and should: provenance that names the
#: wrong module is worse than provenance that shows a discontinuity. Rows
#: written before the move to LazyStats carry
#: ``market_data_hub.regime.estimate``, and that remains true of them.
PROVENANCE_SOURCE = "lazystats.regimes.estimation"


@dataclass(frozen=True)
class SymbolReturns:
    """One symbol's daily returns, ready for the engine.

    Attributes:
        symbol: The bare symbol, as the depot keys it.
        dates: Trading dates, ascending.
        values: Log returns aligned to ``dates``, with gaps already dropped.
    """

    symbol: str
    dates: tuple[str, ...]
    values: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.values)


def symbol_returns(
    instrument: str,
    *,
    start: str = "",
    end: str = "",
) -> SymbolReturns:
    """Load one symbol's daily log returns for fitting.

    The boundary that matters. ``load_returns`` labels its columns canonically
    (``ticker:GLD``) and returns ``None`` for dates where a price was missing;
    the engine wants a bare symbol and a dense series. Both conversions happen
    here, once, so no caller downstream has to remember them — and the symbol
    that comes out is the one :func:`~lazystats.regimes.series.series_key` will
    key on.

    Args:
        instrument: Symbol or canonical id.
        start: Inclusive ISO date, or empty for all available history.
        end: Inclusive ISO date, or empty for the latest available.

    Returns:
        The symbol's returns with missing observations dropped.

    Raises:
        ValueError: The hub returned no usable observations, which means the
            symbol is absent or entirely un-priced over the window. Fitting an
            empty series would produce a model of nothing and persist it.
    """
    symbol = bare_symbol(instrument)
    dataset: ReturnDataset = load_returns([symbol], start=start, end=end, frequency="D")

    if not dataset.instruments:
        raise ValueError(f"the hub returned no series for {symbol!r}")
    column = dataset.instruments[0]

    dates: list[str] = []
    values: list[float] = []
    for row in dataset.rows:
        value = row.get(column)
        if value is None:
            continue
        dates.append(row["date"])
        values.append(float(value))

    if not values:
        raise ValueError(
            f"{symbol!r} has no usable returns between {start or 'the earliest'} "
            f"and {end or 'the latest'} available date"
        )
    return SymbolReturns(symbol=symbol, dates=tuple(dates), values=tuple(values))


def _scalar(value: Any) -> float:
    """The one number inside an arbitrarily nested single-element sequence.

    The engine reports a univariate fit's variance at a nesting depth that
    depends on its covariance type: ``[[v]]`` for a full covariance, ``[v]`` for
    a diagonal one. Indexing at a fixed depth would read a list as a float on
    one branch and crash on the other, so the depth is walked instead.

    Raises:
        ValueError: The sequence bottoms out empty, which means the state has no
            fitted parameter at all rather than a parameter of zero.
    """
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("expected a fitted number, found an empty sequence")
        value = value[0]
    return float(value)


def _as_lists(value: Any) -> Any:
    """Plain Python lists, whatever array type the engine handed back.

    Duck-typed rather than importing numpy: this module must stay importable
    without the ``regimes`` extra, and ``tolist`` is the only thing needed.
    """
    return value.tolist() if hasattr(value, "tolist") else value


def annualized_states(
    means: Any,
    covars: Any,
    labels: Any,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> list[dict[str, Any]]:
    """Per-state annualized mean return and volatility, in state order.

    This is the *interpreted* form of a fit's parameters, and it is what every
    reader downstream actually consumes: :mod:`~lazystats.regimes.tiers` ranks
    on ``annualized_volatility``, and
    :func:`~lazystats.regimes.window_comparison.compare_fits` reads both keys.
    Persisting it, rather than the raw per-period ``means_``/``covars_``, is what
    lets a stored fit be compared without re-deriving anything — and without
    every reader having to know the engine's covariance nesting.

    Args:
        means: One per-period mean per state.
        covars: One per-period variance per state, at any nesting depth.
        labels: The engine's state labels; a state past the end of this list
            falls back to its own index.
        periods_per_year: Periods to annualise over.

    Returns:
        One record per state, carrying ``state``, ``label``,
        ``annualized_mean_return`` and ``annualized_volatility``.

    Raises:
        ValueError: ``means`` and ``covars`` disagree on how many states there
            are, which would pair a state's mean with another state's variance.
    """
    mean_rows = _as_lists(means)
    covar_rows = _as_lists(covars)
    label_list = list(_as_lists(labels) or [])

    if len(mean_rows) != len(covar_rows):
        raise ValueError(
            f"{len(mean_rows)} means and {len(covar_rows)} variances: each state "
            f"would be paired with another state's spread"
        )

    states: list[dict[str, Any]] = []
    for index, (mean_row, covar_row) in enumerate(zip(mean_rows, covar_rows, strict=True)):
        variance = _scalar(covar_row)
        states.append({
            "state": index,
            "label": label_list[index] if index < len(label_list) else str(index),
            "annualized_mean_return": _scalar(mean_row) * periods_per_year,
            # A variance the optimiser drove marginally negative is a zero
            # spread, not an imaginary one.
            "annualized_volatility": math.sqrt(max(variance, 0.0) * periods_per_year),
        })
    return states


def fit_symbol(
    instrument: str,
    *,
    start: str = "",
    end: str = "",
    s_max: int = 3,
    n_starts: int = 20,
    random_state: int = 123,
) -> dict[str, Any]:
    """Fit a regime model to one symbol's daily returns.

    A full refit over whatever history the window allows, rather than applying
    stored parameters — which is what makes it possible to see whether one more
    day changes the model's reading of the past.

    Args:
        instrument: Symbol or canonical id.
        start: Inclusive ISO date, or empty for all available history.
        end: Inclusive ISO date, or empty for the latest available.
        s_max: Most states the search may consider.
        n_starts: Random initialisations per state count.
        random_state: Seed, so a rerun reproduces.

    Returns:
        ``symbol``, the fitted ``diagnostics``, the trading ``dates`` and one
        ``readings`` entry per date. The shape
        :func:`~lazystats.regimes.persist.write_fit` expects.

    Raises:
        ValueError: The symbol has no usable returns over the window.
        ImportError: The ``regimes`` extra is not installed.
    """
    # Imported here, not at module scope: `import lazystats` stays free of
    # numpy, pandas and hmmlearn, which is the whole point of the extras.
    import pandas as pd

    from lazystats.regimes import MSRegimeEngine

    returns = symbol_returns(instrument, start=start, end=end)
    frame = pd.DataFrame(
        {returns.symbol: list(returns.values)},
        index=pd.to_datetime(list(returns.dates)),
    )

    engine = MSRegimeEngine(S_max=s_max, n_starts=n_starts, random_state=random_state)
    run = engine.fit(frame)
    meta = run.meta[returns.symbol]
    n_states = int(meta["S"])

    panel = run.panel
    states = panel[f"{returns.symbol}_state"].astype(int).tolist()
    high_vol = panel[f"{returns.symbol}_highvol"].astype(bool).tolist()
    dates = [str(d.date()) for d in panel.index]

    return {
        "symbol": returns.symbol,
        "dates": dates,
        "readings": [
            {"state": s, "n_states": n_states, "is_high_vol": bool(h)}
            for s, h in zip(states, high_vol, strict=True)
        ],
        "diagnostics": {
            "n_states": n_states,
            "criterion": "bic",
            "bic": float(meta["bic"]),
            "loglik": float(meta["loglik"]),
            "data_start": dates[0],
            "data_end": dates[-1],
            "n_obs": len(dates),
            "labels": meta["labels"],
            "periods_per_year": PERIODS_PER_YEAR,
            # Without this a stored fit cannot be compared against another
            # window's: the comparison ranks states by annualized volatility,
            # and nothing else in the payload carries it.
            "states": annualized_states(meta["means_"], meta["covars_"], meta["labels"]),
        },
    }


def is_production(market_db: str | Path, production_db: str | Path) -> bool:
    """Whether estimates from ``market_db`` may supersede production's series.

    Stated rather than derived. The market database a run reads and the one that
    counts as production are both declared by the caller, so a deployment that
    points its own environment at a different file cannot end up comparing that
    file with itself and concluding it is production.
    """
    return Path(market_db).resolve() == Path(production_db).resolve()


__all__ = [
    "PERIODS_PER_YEAR",
    "PRODUCED_BY",
    "PROVENANCE_SOURCE",
    "SymbolReturns",
    "annualized_states",
    "fit_symbol",
    "is_production",
    "symbol_returns",
]
