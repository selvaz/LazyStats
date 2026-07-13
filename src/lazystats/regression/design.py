"""ReturnDataset → aligned design matrix for the regression fits.

Alignment policy is complete-case (listwise) deletion: only dates where the
dependent AND every regressor are finite survive, and the number of dropped
rows is reported so callers can see how much history the join cost.

numpy is imported lazily (extra: ``lazystats[regression]``) so importing this
module keeps the purity guarantees of the rest of the package.
"""

from __future__ import annotations

from typing import Any

from lazystats.core.returns import series_observations
from lazystats.core.transforms import standardize as _standardize
from lazystats.models import ReturnDataset

__all__ = ["prepare_design"]


def prepare_design(
    dataset: ReturnDataset,
    dependent: str,
    regressors: list[str] | None = None,
    *,
    standardize: bool = False,
) -> dict[str, Any]:
    """Build ``y`` (n,) and ``X`` (n, k) arrays from an aligned dataset.

    ``regressors=None`` uses every other instrument in dataset order.
    ``standardize=True`` z-scores dependent and regressors (column-wise, on
    each column's full non-missing sample) before extraction, so coefficients
    come out as standardized effect sizes.
    """
    if dependent not in dataset.instruments:
        raise ValueError(f"unknown dependent instrument: {dependent!r}")
    if regressors is None:
        regressors = [item for item in dataset.instruments if item != dependent]
    if not regressors:
        raise ValueError("regressors must contain at least one instrument")
    unknown = [item for item in regressors if item not in dataset.instruments]
    if unknown:
        raise ValueError(f"unknown regressor instruments: {', '.join(unknown)}")
    if dependent in regressors:
        raise ValueError("dependent must not appear among the regressors")
    if len(set(regressors)) != len(regressors):
        raise ValueError("regressors must be unique")

    if standardize:
        dataset = _standardize(dataset, instruments=[dependent, *regressors])

    needed = [dependent, *regressors]
    y_values: list[float] = []
    x_rows: list[list[float]] = []
    dates: list[str] = []
    n_dropped = 0
    for date, values in series_observations(dataset):
        if any(item not in values for item in needed):
            n_dropped += 1
            continue
        y_values.append(values[dependent])
        x_rows.append([values[item] for item in regressors])
        dates.append(date)

    minimum = len(regressors) + 2
    if len(y_values) < minimum:
        raise ValueError(
            f"need at least {minimum} aligned observations for {len(regressors)} "
            f"regressor(s), got {len(y_values)}"
        )

    np = _numpy()
    return {
        "y": np.asarray(y_values, dtype=float),
        "X": np.asarray(x_rows, dtype=float),
        "dependent": dependent,
        "regressors": list(regressors),
        "n_obs": len(y_values),
        "n_dropped": n_dropped,
        "dates": (dates[0], dates[-1]),
        "standardized": standardize,
    }


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "lazystats.regression requires the optional dependencies: "
            "pip install 'lazystats[regression]'"
        ) from exc
    return numpy
