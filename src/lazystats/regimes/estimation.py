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

from dataclasses import dataclass
from pathlib import Path

from lazystats.io.datahub import load_returns
from lazystats.models import ReturnDataset
from lazystats.regimes.series import bare_symbol

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


def is_production(market_db: str | Path, production_db: str | Path) -> bool:
    """Whether estimates from ``market_db`` may supersede production's series.

    Stated rather than derived. The market database a run reads and the one that
    counts as production are both declared by the caller, so a deployment that
    points its own environment at a different file cannot end up comparing that
    file with itself and concluding it is production.
    """
    return Path(market_db).resolve() == Path(production_db).resolve()


__all__ = [
    "PRODUCED_BY",
    "PROVENANCE_SOURCE",
    "SymbolReturns",
    "is_production",
    "symbol_returns",
]
