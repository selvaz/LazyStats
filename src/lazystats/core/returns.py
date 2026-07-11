"""Return statistics: volatility, correlation, outliers.

Migrated verbatim from ``lazytools.statistical_analysis`` (plan v3.1 Fase 6):
that implementation — already validated against a live agent — is the
reference behaviour, and its golden numbers are preserved by this package's
tests. Semantics must not drift: LazyTools' wrapper delegates here in the
coexistence release, keeping identical tool signatures.

All functions take an already-loaded :class:`~lazystats.models.ReturnDataset`
and return plain JSON-serialisable dict payloads WITHOUT the LLM-facing
``data`` metadata section, output caps or ``AnalysisResult`` envelope — those
are bridge concerns and stay in LazyTools.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from lazystats.models import ReturnDataset

__all__ = [
    "PERIODS_PER_YEAR",
    "pearson",
    "periods_per_year",
    "return_correlation",
    "return_outliers",
    "return_volatility",
    "series_observations",
    "series_values",
]

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4}


def series_observations(dataset: ReturnDataset) -> list[tuple[str, dict[str, float]]]:
    """Validate rows and discard missing values, preserving date ordering."""
    observations: list[tuple[str, dict[str, float]]] = []
    for row in dataset.rows:
        date = row.get("date")
        if not isinstance(date, str) or not date:
            raise ValueError("return data must contain a non-empty date")
        values: dict[str, float] = {}
        for instrument in dataset.instruments:
            value = row.get(instrument)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric return for {instrument!r} at {date}") from exc
            if math.isfinite(number):
                values[instrument] = number
        observations.append((date, values))
    return observations


def series_values(dataset: ReturnDataset) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {instrument: [] for instrument in dataset.instruments}
    for _, row_values in series_observations(dataset):
        for instrument, value in row_values.items():
            values[instrument].append(value)
    return values


def pearson(pairs: Iterable[tuple[float, float]]) -> float | None:
    values = list(pairs)
    if len(values) < 2:
        return None
    left, right = zip(*values, strict=True)
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in values)
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return _round(numerator / (left_scale * right_scale))


def periods_per_year(frequency: str) -> int:
    try:
        return PERIODS_PER_YEAR[frequency]
    except KeyError as exc:
        raise ValueError("frequency must be one of D, W, M, Q") from exc


def return_volatility(dataset: ReturnDataset, *, frequency: str = "D") -> dict[str, Any]:
    """Sample standard deviation of log returns, annualized by frequency."""
    period_factor = periods_per_year(frequency)
    volatility: dict[str, dict[str, float | int | None]] = {}
    for instrument, values in series_values(dataset).items():
        n = len(values)
        if n < 2:
            volatility[instrument] = {
                "observations": n,
                "mean_log_return": None,
                "period_volatility": None,
                "annualized_volatility": None,
            }
            continue
        sigma = statistics.stdev(values)
        volatility[instrument] = {
            "observations": n,
            "mean_log_return": _round(statistics.fmean(values)),
            "period_volatility": _round(sigma),
            "annualized_volatility": _round(sigma * math.sqrt(period_factor)),
        }
    return {
        "metric": "sample standard deviation of log returns",
        "frequency": frequency,
        "periods_per_year": period_factor,
        "volatility": volatility,
    }


def return_correlation(
    dataset: ReturnDataset, *, frequency: str = "D", min_periods: int = 2
) -> dict[str, Any]:
    """Pairwise Pearson correlation over the dates where both series exist."""
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    observations = series_observations(dataset)
    correlation: dict[str, dict[str, float | None]] = {}
    pair_counts: dict[str, dict[str, int]] = {}
    for left in dataset.instruments:
        correlation[left] = {}
        pair_counts[left] = {}
        for right in dataset.instruments:
            paired = [
                (values[left], values[right])
                for _, values in observations
                if left in values and right in values
            ]
            pair_counts[left][right] = len(paired)
            correlation[left][right] = pearson(paired) if len(paired) >= min_periods else None
    return {
        "metric": "Pearson correlation of log returns",
        "frequency": frequency,
        "min_periods": min_periods,
        "correlation": correlation,
        "pairwise_observations": pair_counts,
    }


def return_outliers(
    dataset: ReturnDataset, *, frequency: str = "D", threshold: float = 2.0
) -> dict[str, Any]:
    """Full-period z-score outliers, sorted by severity then date/instrument.

    Returns ALL outliers — result caps for LLM context safety are the
    bridge's concern (LazyTools applies default/hard caps before emitting).
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a finite value greater than zero")
    series = series_values(dataset)
    z_scores: dict[str, tuple[float, float] | None] = {}
    for instrument, values in series.items():
        if len(values) < 2:
            z_scores[instrument] = None
            continue
        sigma = statistics.stdev(values)
        z_scores[instrument] = None if sigma == 0 else (statistics.fmean(values), sigma)

    outliers: list[dict[str, Any]] = []
    for date, observation_values in series_observations(dataset):
        for instrument in dataset.instruments:
            value = observation_values.get(instrument)
            params = z_scores[instrument]
            if value is None or params is None:
                continue
            mean, sigma = params
            z_score = (value - mean) / sigma
            if abs(z_score) >= threshold:
                outliers.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "log_return": _round(value),
                        "z_score": _round(z_score),
                        "direction": "positive" if z_score > 0 else "negative",
                    }
                )

    outliers.sort(key=lambda item: (-abs(float(item["z_score"])), item["date"], item["instrument"]))
    return {
        "metric": "period z-score of log returns",
        "frequency": frequency,
        "threshold": threshold,
        "comparison": "abs(z_score) >= threshold",
        "total_outliers": len(outliers),
        "outliers": outliers,
    }


def _round(value: float) -> float:
    return round(float(value), 10)
