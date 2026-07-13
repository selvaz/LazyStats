"""Pure statistics — the reference behaviour behind LazyTools' statistical
tools (plan v3.1 Fase 6). Stdlib-only; loading data is ``lazystats.io``'s job,
serialization/budgeting for LLMs is LazyTools' job."""

from lazystats.core.returns import (
    PERIODS_PER_YEAR,
    pearson,
    periods_per_year,
    return_correlation,
    return_outliers,
    return_volatility,
    series_observations,
    series_values,
)
from lazystats.core.transforms import demean, standardize

__all__ = [
    "PERIODS_PER_YEAR",
    "demean",
    "pearson",
    "periods_per_year",
    "return_correlation",
    "return_outliers",
    "return_volatility",
    "series_observations",
    "series_values",
    "standardize",
]
