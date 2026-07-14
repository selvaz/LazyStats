"""OLS / Ridge / Lasso wrappers (extra: ``lazystats[regression]``).

Nothing is re-implemented: OLS inference comes from statsmodels (robust
covariances included), Ridge/Lasso and their cross-validated alpha selection
from scikit-learn. This module only adapts :class:`ReturnDataset` inputs and
flattens results into plain JSON-serialisable dicts — no residual or fitted
series ever leaves a fit, only coefficients and diagnostics. LLM envelope,
output caps and ``data`` metadata are bridge concerns and stay in LazyTools.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from lazystats.models import ReturnDataset
from lazystats.regression.design import prepare_design

__all__ = ["fit_lasso", "fit_ols", "fit_ridge"]

_OLS_COV_TYPES = {"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"}


def _resolve_dataset(dataset: ReturnDataset | None, data_key: str) -> ReturnDataset:
    """Return the ``ReturnDataset`` to fit, from a depot ``data_key`` or inline.

    ``data_key`` is the store handle written by
    ``lazystats.regimes.load_from_datahub`` (a returns matrix
    ``{"Y", "columns", "index"}``). Reusing that shared depot is what lets an
    agent load returns once and then fit regimes AND regressions off the same
    handle, without any array ever crossing the tool boundary. The depot lives
    in the ``regimes`` tier, imported lazily so ``regression`` stays usable
    without it when callers pass an explicit ``dataset``.
    """
    if not data_key:
        if dataset is None:
            raise ValueError("provide either data_key (a returns store handle) or dataset")
        return dataset
    try:
        from lazystats.regimes.tools import _sread
    except ImportError as exc:  # pragma: no cover - needs the extra absent
        raise ImportError(
            "the data_key path shares the returns depot from lazystats[regimes]: "
            "pip install 'lazystats[regimes]'"
        ) from exc
    stored = _sread(data_key)  # {"Y": ndarray(T,k), "columns": [...], "index": [...]}
    matrix, columns, index = stored["Y"], list(stored["columns"]), list(stored["index"])
    rows: list[dict[str, Any]] = []
    for i, stamp in enumerate(index):
        row: dict[str, Any] = {"date": str(stamp)[:10]}
        for j, column in enumerate(columns):
            value = float(matrix[i][j])
            row[column] = value if math.isfinite(value) else None
        rows.append(row)
    return ReturnDataset(
        instruments=columns, rows=rows, metadata={"source": "depot", "data_key": data_key}
    )


def fit_ols(
    dataset: ReturnDataset | None = None,
    dependent: str = "",
    regressors: Annotated[
        list[str] | None,
        "Right-hand series names (columns of the loaded returns). Omit/empty to use "
        "every other loaded column.",
    ] = None,
    *,
    data_key: Annotated[
        str,
        "Store key from load_from_datahub(data_key=...) (or load_time_series). "
        "Preferred for LLM use — avoids passing return arrays in tool calls. "
        "Alternative to dataset=.",
    ] = "",
    add_constant: bool = True,
    cov: str = "nonrobust",
    hac_lags: int | None = None,
    ci_level: float = 0.95,
    standardize: bool = False,
) -> dict[str, Any]:
    """Ordinary least squares via ``statsmodels.api.OLS``.

    ``dataset``/``dependent``/``regressors`` stay positional for backward
    compatibility with existing callers; pass ``data_key`` instead of
    ``dataset`` to read returns from the shared depot also used by
    ``lazystats.regimes`` (``load_from_datahub``) rather than an inline
    ``ReturnDataset``. Univariate regression is simply the 1-regressor case.
    ``cov`` selects the covariance estimator: ``nonrobust``, ``HC0``–``HC3``
    (heteroskedasticity-robust) or ``HAC`` (Newey-West; ``hac_lags=None`` uses
    the common ``floor(4 * (n/100) ** (2/9))`` rule).
    """
    if not dependent:
        raise ValueError("dependent is required")
    if cov not in _OLS_COV_TYPES:
        raise ValueError(f"cov must be one of {', '.join(sorted(_OLS_COV_TYPES))}")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be strictly between 0 and 1")
    if hac_lags is not None and hac_lags < 1:
        raise ValueError("hac_lags must be at least 1 when provided")

    design = prepare_design(
        _resolve_dataset(dataset, data_key), dependent, regressors, standardize=standardize
    )
    sm, stattools = _import_statsmodels()

    exog = design["X"]
    names: list[str] = list(design["regressors"])
    if add_constant:
        exog = sm.add_constant(exog, has_constant="add")
        names = ["const", *names]

    model = sm.OLS(design["y"], exog)
    if cov == "nonrobust":
        result = model.fit()
        used_lags = None
    elif cov == "HAC":
        used_lags = hac_lags or max(1, math.floor(4 * (design["n_obs"] / 100) ** (2 / 9)))
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": used_lags})
    else:
        result = model.fit(cov_type=cov)
        used_lags = None

    conf = result.conf_int(alpha=1 - ci_level)
    coefficients = {
        name: {
            "coef": _safe(result.params[i]),
            "std_err": _safe(result.bse[i]),
            "t_stat": _safe(result.tvalues[i]),
            "p_value": _safe(result.pvalues[i]),
            "ci_low": _safe(conf[i][0]),
            "ci_high": _safe(conf[i][1]),
        }
        for i, name in enumerate(names)
    }

    residuals = result.resid
    jb_stat, jb_pvalue, jb_skew, jb_kurtosis = stattools.jarque_bera(residuals)
    return {
        "model": "ols",
        "dependent": design["dependent"],
        "regressors": design["regressors"],
        "n_obs": design["n_obs"],
        "n_dropped": design["n_dropped"],
        "dates": list(design["dates"]),
        "standardized": design["standardized"],
        "add_constant": add_constant,
        "cov_type": cov,
        "hac_lags": used_lags,
        "ci_level": ci_level,
        "coefficients": coefficients,
        "r_squared": _safe(result.rsquared),
        "adj_r_squared": _safe(result.rsquared_adj),
        "f_stat": _safe(result.fvalue),
        "f_pvalue": _safe(result.f_pvalue),
        "aic": _safe(result.aic),
        "bic": _safe(result.bic),
        "durbin_watson": _safe(stattools.durbin_watson(residuals)),
        "condition_number": _safe(result.condition_number),
        "residual_diagnostics": {
            "jarque_bera": _safe(jb_stat),
            "jarque_bera_pvalue": _safe(jb_pvalue),
            "skew": _safe(jb_skew),
            "kurtosis": _safe(jb_kurtosis),
            "resid_mean": _safe(residuals.mean()),
            "resid_std": _safe(residuals.std(ddof=1)),
        },
    }


def fit_ridge(
    dataset: ReturnDataset | None = None,
    dependent: str = "",
    regressors: Annotated[
        list[str] | None,
        "Right-hand series names (columns of the loaded returns). Omit/empty to use "
        "every other loaded column.",
    ] = None,
    *,
    data_key: Annotated[
        str, "Store key from load_from_datahub(data_key=...). Alternative to dataset=."
    ] = "",
    alpha: float | None = None,
    alphas: list[float] | None = None,
    cv_folds: int = 5,
    standardize: bool = True,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    """Ridge regression via scikit-learn; ``alpha=None`` cross-validates.

    ``dataset``/``dependent``/``regressors`` stay positional for backward
    compatibility; pass ``data_key`` instead of ``dataset`` to read returns
    from the shared depot also used by ``lazystats.regimes``.
    """
    if not dependent:
        raise ValueError("dependent is required")
    return _fit_regularized(
        "ridge",
        _resolve_dataset(dataset, data_key),
        dependent,
        regressors,
        alpha=alpha,
        alphas=alphas,
        cv_folds=cv_folds,
        standardize=standardize,
        fit_intercept=fit_intercept,
    )


def fit_lasso(
    dataset: ReturnDataset | None = None,
    dependent: str = "",
    regressors: Annotated[
        list[str] | None,
        "Right-hand series names (columns of the loaded returns). Omit/empty to use "
        "every other loaded column.",
    ] = None,
    *,
    data_key: Annotated[
        str, "Store key from load_from_datahub(data_key=...). Alternative to dataset=."
    ] = "",
    alpha: float | None = None,
    cv_folds: int = 5,
    standardize: bool = True,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    """Lasso regression via scikit-learn; ``alpha=None`` cross-validates.

    ``dataset``/``dependent``/``regressors`` stay positional for backward
    compatibility; pass ``data_key`` instead of ``dataset`` to read returns
    from the shared depot also used by ``lazystats.regimes``.
    """
    if not dependent:
        raise ValueError("dependent is required")
    return _fit_regularized(
        "lasso",
        _resolve_dataset(dataset, data_key),
        dependent,
        regressors,
        alpha=alpha,
        alphas=None,
        cv_folds=cv_folds,
        standardize=standardize,
        fit_intercept=fit_intercept,
    )


def _fit_regularized(
    kind: str,
    dataset: ReturnDataset,
    dependent: str,
    regressors: list[str] | None,
    *,
    alpha: float | None,
    alphas: list[float] | None,
    cv_folds: int,
    standardize: bool,
    fit_intercept: bool,
) -> dict[str, Any]:
    if alpha is not None and alpha <= 0:
        raise ValueError("alpha must be greater than zero when provided")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    design = prepare_design(dataset, dependent, regressors)
    np, linear_model, preprocessing = _import_sklearn()

    y = design["y"]
    x_raw = design["X"]
    if design["n_obs"] < cv_folds and alpha is None:
        raise ValueError(
            f"cross-validation needs at least cv_folds={cv_folds} observations, "
            f"got {design['n_obs']}"
        )

    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler()
        x_fit = scaler.fit_transform(x_raw)
    else:
        x_fit = x_raw

    cv_r_squared: float | None = None
    if alpha is not None:
        estimator = (
            linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept)
            if kind == "ridge"
            else linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10_000)
        )
        estimator.fit(x_fit, y)
        chosen_alpha = float(alpha)
        selection = "fixed"
    elif kind == "ridge":
        grid = np.asarray(alphas, dtype=float) if alphas else np.logspace(-4, 4, 50)
        estimator = linear_model.RidgeCV(alphas=grid, fit_intercept=fit_intercept, cv=cv_folds)
        estimator.fit(x_fit, y)
        chosen_alpha = float(estimator.alpha_)
        cv_r_squared = _safe(getattr(estimator, "best_score_", None))
        selection = "cv"
    else:
        # default alpha grid (100 values) — the keyword spelling changed across
        # sklearn versions (n_alphas -> alphas), so rely on the default.
        estimator = linear_model.LassoCV(
            cv=cv_folds, fit_intercept=fit_intercept, max_iter=10_000
        )
        estimator.fit(x_fit, y)
        chosen_alpha = float(estimator.alpha_)
        selection = "cv"

    fitted_coefs = np.asarray(estimator.coef_, dtype=float)
    fitted_intercept = float(estimator.intercept_) if fit_intercept else 0.0

    if scaler is not None:
        scale = np.asarray(scaler.scale_, dtype=float)
        center = np.asarray(scaler.mean_, dtype=float)
        original_coefs = fitted_coefs / scale
        original_intercept = fitted_intercept - float((fitted_coefs * center / scale).sum())
        standardized_coefficients = {
            name: _safe(value)
            for name, value in zip(design["regressors"], fitted_coefs, strict=True)
        }
    else:
        original_coefs = fitted_coefs
        original_intercept = fitted_intercept
        standardized_coefficients = None

    payload: dict[str, Any] = {
        "model": kind,
        "dependent": design["dependent"],
        "regressors": design["regressors"],
        "n_obs": design["n_obs"],
        "n_dropped": design["n_dropped"],
        "dates": list(design["dates"]),
        "alpha": _safe(chosen_alpha),
        "alpha_selection": selection,
        "cv_folds": cv_folds if selection == "cv" else None,
        "standardized": standardize,
        "fit_intercept": fit_intercept,
        "intercept": _safe(original_intercept) if fit_intercept else None,
        "coefficients": {
            name: _safe(value)
            for name, value in zip(design["regressors"], original_coefs, strict=True)
        },
        "standardized_coefficients": standardized_coefficients,
        "r_squared": _safe(float(estimator.score(x_fit, y))),
        "cv_r_squared": cv_r_squared,
    }
    if kind == "lasso":
        nonzero = [
            name
            for name, value in zip(design["regressors"], fitted_coefs, strict=True)
            if value != 0.0
        ]
        payload["n_nonzero"] = len(nonzero)
        payload["selected_regressors"] = nonzero
    return payload


def _import_statsmodels() -> tuple[Any, Any]:
    try:
        import statsmodels.api as sm
        from statsmodels.stats import stattools
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "lazystats.regression requires the optional dependencies: "
            "pip install 'lazystats[regression]'"
        ) from exc
    return sm, stattools


def _import_sklearn() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from sklearn import linear_model, preprocessing
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "lazystats.regression requires the optional dependencies: "
            "pip install 'lazystats[regression]'"
        ) from exc
    return np, linear_model, preprocessing


def _safe(value: Any) -> float | None:
    """Round to 10 decimals; NaN/inf/None become None (JSON-safe)."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 10)
