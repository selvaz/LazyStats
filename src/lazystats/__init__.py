"""lazystats — pure statistical library of the LazyBridge ecosystem.

Plan v3.1, Fase 6: the single home for statistical computation. Pure Python
(stdlib-only core, zero hard dependencies), usable from notebooks and services
without importing lazybridge — all LLM tool wrapping lives in LazyTools
(``lazytools.statistical_analysis``), which these functions are the reference
behaviour for.

Layout:
    core/    pure statistics (volatility, correlation, outliers, helpers)
    models/  shared data shapes (ReturnDataset)
    io/      datahub.py (hub-backed loading, lazy import), depot.py (SQLite
             result depot with provenance), local.py (notebook-only loaders —
             NEVER exposed as LLM tools)

    regimes/ LazyHMM migrated here (extra: lazystats[regimes]); the lazyhmm
             package is a coexistence shim delegating to this subpackage.

LazyRay is frozen and migrates only after numeric + depot equivalence.
"""

from lazystats.core import (
    PERIODS_PER_YEAR,
    demean,
    pearson,
    periods_per_year,
    return_correlation,
    return_outliers,
    return_volatility,
    series_observations,
    series_values,
    standardize,
)
from lazystats.models import ReturnDataset

__version__ = "0.2.0"

__all__ = [
    "ReturnDataset",
    "return_volatility",
    "return_correlation",
    "return_outliers",
    "series_observations",
    "series_values",
    "standardize",
    "demean",
    "pearson",
    "periods_per_year",
    "PERIODS_PER_YEAR",
    "__version__",
]
