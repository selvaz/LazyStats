# -*- coding: utf-8 -*-
"""
lazystats.regimes.datasources.datahub — market-data-hub → LazyHMM depot loader
====================================================================
A thin, depot-aware data-source loader that pulls a returns matrix from the
``market-data-hub`` package and stores it under a ``data_key`` in the LazyHMM
depot using the EXACT payload shape that
:func:`lazystats.regimes.tools.load_time_series` writes::

    {"Y": <float ndarray (T, k)>, "columns": [...], "index": [str, ...]}

so the existing ``fit_regimes(data_key=...)`` tool (and ``RegimeEngine``)
consume it with no further glue.

``market-data-hub`` is an *optional* dependency: it is imported lazily inside
``load_from_datahub``.  Install it with the ``datahub`` extra::

    pip install 'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Reuse the SAME store-write helper that load_time_series uses so the data
# lands in the in-process cache / SQLite depot / lazybridge Store identically.
from ..tools import _swrite

# Module-level handle for the market-data-hub entry point. It is resolved
# lazily by ``_get_extract_returns`` on first use and cached here. Exposing it
# as a module attribute is what lets tests patch the dependency cheaply via
# ``monkeypatch.setattr(lazystats.regimes.datasources.datahub, "extract_returns", ...)``
# — i.e. patch where the name is looked up.
extract_returns = None  # type: ignore[assignment]


def _get_extract_returns():
    """Return the market-data-hub ``extract_returns`` callable (lazy import).

    Honours a module-level override (set by tests) before importing, then
    falls back to the top-level package and finally the ``extract`` submodule.
    """
    if extract_returns is not None:
        return extract_returns
    try:
        from market_data_hub.extract import extract_returns as _er  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "market-data-hub is required for load_from_datahub(). It is a "
            "private, git-installed package: pip install "
            "'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'"
        ) from exc
    return _er


def load_from_datahub(
    symbols: Union[str, Sequence[str]],
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    frequency: str = "W",
    field: str = "adj_close",
    fillna: str = "none",
    data_key: str = "datahub",
    db_path: Optional[str] = None,
    store: Optional[Any] = None,  # reserved for an explicit store override
) -> dict:
    """Pull a returns matrix from ``market-data-hub`` and store it in the depot.

    Calls ``market_data_hub.extract_returns(...)`` to obtain a wide log-returns
    DataFrame (DatetimeIndex, one column per symbol), then writes the SAME
    payload as :func:`lazystats.regimes.tools.load_time_series` under ``data_key`` so
    ``fit_regimes(data_key=...)`` can consume it directly.

    Args:
        symbols: One symbol, a list of symbols, or a comma/semicolon-separated
            string (e.g. 'SPY', ['SPY', 'TLT'], or 'SPY,TLT,GLD'). The delimited
            string form is the one to use across a tool/MCP boundary, where a
            JSON list cannot be passed.
        start: ISO start date 'YYYY-MM-DD' (or None for earliest available).
        end: ISO end date 'YYYY-MM-DD' (or None for latest available).
        frequency: Resampling frequency passed to extract_returns
            ('D', 'W', 'M', ...). Default 'W' (weekly), the shape LazyHMM expects.
        field: Price field to take returns of (e.g. 'adj_close').
        fillna: Missing-value policy forwarded to extract_returns
            ('none', 'ffill', ...).
        data_key: Depot key under which the returns matrix is stored. Pass this
            to fit_regimes(data_key=...).
        db_path: Optional path to the market-data-hub DuckDB file (forwarded to
            extract_returns); None uses its default.
        store: Reserved for an explicit store override (unused — the loader
            writes through the standard LazyHMM depot via _swrite).

    Returns:
        dict with keys:
            data_key (str): key to pass to fitting tools.
            n_rows (int): number of timesteps loaded.
            n_cols (int): number of series (= number of symbols returned).
            columns (list[str]): the returned column names.
            date_range (list[str]): [first_date, last_date] as ISO-ish strings,
                or [] if no rows were returned.
            source (str): always 'market-data-hub'.
            frequency (str): the frequency requested.
            field (str): the price field used.

    Raises:
        ImportError: if the ``market-data-hub`` package is not installed.
    """
    # ── Normalize a delimited string into a symbol list ─────────────────────
    # Callers reaching this through an MCP/tool boundary can only pass a scalar
    # string (a JSON list arrives wrapped as a single element), so accept
    # comma- or semicolon-separated symbols in one string, e.g. "SPY,TLT,GLD".
    # A bare single token (no separator) is left untouched to preserve the
    # exact single-symbol behaviour extract_returns already handles.
    if isinstance(symbols, str) and ("," in symbols or ";" in symbols):
        symbols = [s.strip() for s in symbols.replace(";", ",").split(",") if s.strip()]

    # ── Lazy, optional import (mirrors the yfinance pattern in lazystats.regimes.db) ──
    # Resolved via the module-level handle so tests can monkeypatch
    # ``lazystats.regimes.datasources.datahub.extract_returns``.
    _extract_returns = _get_extract_returns()

    df, _meta = _extract_returns(
        symbols,
        start=start,
        end=end,
        frequency=frequency,
        field=field,
        fillna=fillna,
        db_path=db_path,
    )

    # Guarantee a finite matrix for downstream HMM fitting. With multi-symbol
    # staggered histories (or partial gaps) and fillna="none", extract_returns
    # only drops all-NaN rows — a single NaN in one column would otherwise reach
    # fit_regimes(data_key=...) and break the fit. Drop any row with a missing
    # value, and fail loudly rather than store an unusable (empty) matrix.
    df = df.dropna(how="any")
    if df.shape[0] == 0:
        req = [symbols] if isinstance(symbols, str) else list(symbols)
        raise ValueError(
            "load_from_datahub: no rows with complete data across all requested "
            f"symbols {req} after dropping missing values. Widen the date range, "
            "use a coarser frequency, or pass fillna='ffill'."
        )

    # ── Build the SAME payload load_time_series stores ──────────────────────
    Y: np.ndarray = df.values.astype(float)
    columns: List[str] = list(df.columns)
    index: List[str] = [str(i) for i in df.index]

    _swrite(data_key, {"Y": Y, "columns": columns, "index": index})

    # ── Provenance / summary (mirrors load_time_series' return + extras) ────
    date_range: List[str] = []
    if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
        date_range = [str(df.index[0])[:10], str(df.index[-1])[:10]] if len(df.index) else []
    elif len(df.index) > 0:
        date_range = [str(df.index[0]), str(df.index[-1])]

    result: Dict[str, Any] = {
        "data_key": data_key,
        "n_rows": int(Y.shape[0]),
        "n_cols": int(Y.shape[1]) if Y.ndim > 1 else (1 if Y.size else 0),
        "columns": columns,
        "date_range": date_range,
        "source": "market-data-hub",
        "frequency": frequency,
        "field": field,
    }
    return result
