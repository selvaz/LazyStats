"""market-data-hub loader — the DEFAULT data path (plan v3.1 Fase 6).

Mirrors the loading semantics of LazyTools'
``MarketDataHubStatisticsBackend``: canonical lazydatacore ids, direct
``extract.extract_returns`` call (never the truncating agent JSON tool —
truncated series would silently corrupt the statistics), full metadata.

``market_data_hub`` is a private git-installed package, imported lazily; a
missing install raises a clear ImportError with the pip hint.
"""

from __future__ import annotations

import math
from typing import Any

from lazystats.models import ReturnDataset

__all__ = ["load_returns"]


def load_returns(
    instruments: str | list[str],
    *,
    start: str = "",
    end: str = "",
    frequency: str = "D",
) -> ReturnDataset:
    """Load log returns for one or more instruments from the hub.

    instruments: comma-separated string or list; bare symbols are canonicalised
    to ``ticker:<SYM>``. Only the ticker domain is supported.
    """
    if frequency not in ("D", "W", "M", "Q"):
        raise ValueError("frequency must be one of D, W, M, Q")
    try:
        from market_data_hub import extract
        from market_data_hub.lazydatacore import Domain, InstrumentId
    except ImportError as exc:  # pragma: no cover - exercised without the hub
        raise ImportError(
            "lazystats.io.datahub requires market-data-hub (a private, "
            "git-installed package): pip install "
            "'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'"
        ) from exc

    raw = instruments.split(",") if isinstance(instruments, str) else list(instruments)
    parsed: list[Any] = []
    seen: set[str] = set()
    for item in raw:
        text = item.strip()
        if not text:
            continue
        iid = InstrumentId.parse(text if ":" in text else f"ticker:{text}")
        if iid.domain is not Domain.TICKER:
            raise ValueError(f"only ticker instruments are supported, got {iid}")
        if str(iid) in seen:
            raise ValueError(f"duplicate instrument {iid}")
        seen.add(str(iid))
        parsed.append(iid)
    if not parsed:
        raise ValueError("no instruments provided")

    symbols = [iid.key for iid in parsed]
    frame, metadata = extract.extract_returns(
        symbols, start=start or None, end=end or None, frequency=frequency
    )
    labels = {iid.key: str(iid) for iid in parsed}
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        date = index.date() if hasattr(index, "date") else index
        entry: dict[str, Any] = {"date": str(date)}
        for symbol in symbols:
            value = row.get(symbol)
            entry[labels[symbol]] = (
                float(value) if value is not None and _is_finite(value) else None
            )
        rows.append(entry)

    meta = dict(metadata or {})
    meta.update(
        instruments=[str(iid) for iid in parsed],
        requested_start=start,
        requested_end=end,
        frequency=frequency,
        return_kind="log",
        source="market-data-hub",
    )
    return ReturnDataset(
        instruments=[str(iid) for iid in parsed], rows=rows, metadata=meta
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
