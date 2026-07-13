"""Callable dataset post-transforms: standardize (z-score) and demean.

Applied AFTER loading/extraction (the hub already provides the base
transforms level/log_return/pct_change/diff at fetch time): these operate on
an already-loaded :class:`~lazystats.models.ReturnDataset` and return a new
one, so any downstream statistic — correlation, volatility, regression — can
consume standardized series without touching the loaders.

Stdlib-only, like the rest of ``core``. Missing observations stay ``None``;
column statistics are computed on each column's non-missing sample.
"""

from __future__ import annotations

import statistics
from typing import Any

from lazystats.core.returns import series_observations, series_values
from lazystats.models import ReturnDataset

__all__ = ["demean", "standardize"]


def standardize(
    dataset: ReturnDataset, *, instruments: list[str] | None = None
) -> ReturnDataset:
    """Z-score each selected column: (x - mean) / sample stdev.

    Raises ValueError when a selected column has fewer than 2 observations or
    zero variance — a silently degenerate standardized column would corrupt
    any statistic computed on it.
    """
    return _column_transform(dataset, "standardize", instruments)


def demean(
    dataset: ReturnDataset, *, instruments: list[str] | None = None
) -> ReturnDataset:
    """Subtract each selected column's mean from its observations."""
    return _column_transform(dataset, "demean", instruments)


def _column_transform(
    dataset: ReturnDataset, kind: str, instruments: list[str] | None
) -> ReturnDataset:
    selected = _validate_selection(dataset, instruments)
    values = series_values(dataset)

    params: dict[str, tuple[float, float]] = {}
    for instrument in selected:
        sample = values[instrument]
        if len(sample) < 2:
            raise ValueError(
                f"cannot {kind} {instrument!r}: need at least 2 observations, got {len(sample)}"
            )
        mean = statistics.fmean(sample)
        sigma = statistics.stdev(sample) if kind == "standardize" else 1.0
        if sigma == 0:
            raise ValueError(f"cannot standardize {instrument!r}: zero variance")
        params[instrument] = (mean, sigma)

    rows: list[dict[str, Any]] = []
    for (date, observed), original in zip(
        series_observations(dataset), dataset.rows, strict=True
    ):
        row: dict[str, Any] = {"date": date}
        for instrument in dataset.instruments:
            if instrument not in original and instrument not in observed:
                continue
            value = observed.get(instrument)
            if value is None or instrument not in params:
                row[instrument] = value
                continue
            mean, sigma = params[instrument]
            row[instrument] = _round((value - mean) / sigma)
        rows.append(row)

    post = dict(dataset.metadata.get("post_transforms", {}))
    post.update({instrument: kind for instrument in selected})
    return ReturnDataset(
        instruments=list(dataset.instruments),
        rows=rows,
        metadata={**dataset.metadata, "post_transforms": post},
    )


def _validate_selection(
    dataset: ReturnDataset, instruments: list[str] | None
) -> list[str]:
    if instruments is None:
        return list(dataset.instruments)
    if not instruments:
        raise ValueError("instruments must not be an empty selection")
    unknown = [item for item in instruments if item not in dataset.instruments]
    if unknown:
        raise ValueError(f"unknown instruments: {', '.join(unknown)}")
    if len(set(instruments)) != len(instruments):
        raise ValueError("instruments selection must be unique")
    return list(instruments)


def _round(value: float) -> float:
    return round(float(value), 10)
