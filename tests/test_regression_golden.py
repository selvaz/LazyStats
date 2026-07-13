"""Golden tests for lazystats.regression (extra: lazystats[regression]).

The fixture is deterministic: x1/x2/noise come from fixed trigonometric
sequences, y = 0.001 + 2*x1 - 0.5*x2 + eps, plus an irrelevant x3 regressor
and a few None holes. Univariate OLS is cross-checked against the closed-form
slope cov(x, y)/var(x) computed independently with the stdlib.
"""

from __future__ import annotations

import json
import math
import statistics

import pytest

pytest.importorskip("numpy")
pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")

from lazystats import ReturnDataset
from lazystats.regression import fit_lasso, fit_ols, fit_ridge, prepare_design

TRUE_CONST = 0.001
TRUE_B1 = 2.0
TRUE_B2 = -0.5


def _make_dataset(n: int = 60) -> ReturnDataset:
    rows = []
    for i in range(n):
        x1 = 0.01 * math.sin(0.7 * i)
        x2 = 0.01 * math.cos(1.3 * i)
        x3 = 0.01 * math.sin(2.1 * i + 0.5)  # irrelevant: true coefficient 0
        eps = 0.0005 * math.sin(2.9 * i + 1.0)
        y = TRUE_CONST + TRUE_B1 * x1 + TRUE_B2 * x2 + eps
        rows.append(
            {
                "date": f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}",
                "ticker:Y": round(y, 12),
                "ticker:X1": round(x1, 12),
                "ticker:X2": round(x2, 12),
                "ticker:X3": round(x3, 12),
            }
        )
    rows[5]["ticker:X1"] = None  # complete-case holes
    rows[11]["ticker:Y"] = None
    return ReturnDataset(
        instruments=["ticker:Y", "ticker:X1", "ticker:X2", "ticker:X3"],
        rows=rows,
        metadata={"source": "market-data-hub"},
    )


@pytest.fixture()
def dataset() -> ReturnDataset:
    return _make_dataset()


def test_prepare_design_alignment(dataset: ReturnDataset) -> None:
    design = prepare_design(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"])
    assert design["n_obs"] == 58
    assert design["n_dropped"] == 2
    assert design["y"].shape == (58,)
    assert design["X"].shape == (58, 2)
    assert design["dates"][0] == "2024-01-01"


def test_prepare_design_validation(dataset: ReturnDataset) -> None:
    with pytest.raises(ValueError, match="unknown dependent"):
        prepare_design(dataset, "ticker:NOPE")
    with pytest.raises(ValueError, match="unknown regressor"):
        prepare_design(dataset, "ticker:Y", ["ticker:NOPE"])
    with pytest.raises(ValueError, match="must not appear"):
        prepare_design(dataset, "ticker:Y", ["ticker:Y"])
    with pytest.raises(ValueError, match="unique"):
        prepare_design(dataset, "ticker:Y", ["ticker:X1", "ticker:X1"])
    with pytest.raises(ValueError, match="at least one"):
        prepare_design(dataset, "ticker:Y", [])


def test_prepare_design_requires_enough_observations() -> None:
    tiny = ReturnDataset(
        instruments=["ticker:Y", "ticker:X1"],
        rows=[
            {"date": "2024-01-01", "ticker:Y": 0.01, "ticker:X1": 0.02},
            {"date": "2024-01-02", "ticker:Y": 0.02, "ticker:X1": 0.01},
        ],
    )
    with pytest.raises(ValueError, match="at least 3 aligned observations"):
        prepare_design(tiny, "ticker:Y", ["ticker:X1"])


def test_ols_multivariate_recovers_coefficients(dataset: ReturnDataset) -> None:
    out = fit_ols(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"])
    assert out["model"] == "ols"
    assert out["n_obs"] == 58
    assert out["coefficients"]["const"]["coef"] == pytest.approx(TRUE_CONST, abs=2e-4)
    assert out["coefficients"]["ticker:X1"]["coef"] == pytest.approx(TRUE_B1, abs=0.05)
    assert out["coefficients"]["ticker:X2"]["coef"] == pytest.approx(TRUE_B2, abs=0.05)
    assert out["r_squared"] > 0.99
    assert out["coefficients"]["ticker:X1"]["p_value"] < 1e-6
    x1 = out["coefficients"]["ticker:X1"]
    assert x1["ci_low"] < TRUE_B1 < x1["ci_high"]
    assert out["f_stat"] > 100
    assert out["durbin_watson"] is not None
    assert out["residual_diagnostics"]["resid_mean"] == pytest.approx(0.0, abs=1e-9)
    json.dumps(out)  # payload must be JSON-serialisable as-is


def test_ols_univariate_matches_closed_form(dataset: ReturnDataset) -> None:
    """Independent cross-check: slope = cov(x, y)/var(x) via the stdlib."""
    out = fit_ols(dataset, "ticker:Y", ["ticker:X1"])
    pairs = [
        (row["ticker:X1"], row["ticker:Y"])
        for row in dataset.rows
        if row["ticker:X1"] is not None and row["ticker:Y"] is not None
    ]
    xs, ys = zip(*pairs, strict=True)
    slope = statistics.covariance(xs, ys) / statistics.variance(xs)
    intercept = statistics.fmean(ys) - slope * statistics.fmean(xs)
    corr = statistics.correlation(xs, ys)
    assert out["coefficients"]["ticker:X1"]["coef"] == pytest.approx(slope)
    assert out["coefficients"]["const"]["coef"] == pytest.approx(intercept)
    assert out["r_squared"] == pytest.approx(corr**2)


def test_ols_hac_changes_standard_errors(dataset: ReturnDataset) -> None:
    plain = fit_ols(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"])
    hac = fit_ols(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"], cov="HAC")
    assert hac["cov_type"] == "HAC"
    assert hac["hac_lags"] >= 1
    assert hac["coefficients"]["ticker:X1"]["coef"] == plain["coefficients"]["ticker:X1"]["coef"]
    assert (
        hac["coefficients"]["ticker:X1"]["std_err"]
        != plain["coefficients"]["ticker:X1"]["std_err"]
    )


def test_ols_no_constant_and_standardize(dataset: ReturnDataset) -> None:
    out = fit_ols(dataset, "ticker:Y", ["ticker:X1"], add_constant=False)
    assert "const" not in out["coefficients"]
    std = fit_ols(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"], standardize=True)
    assert std["standardized"] is True
    # standardized effect size: beta * std(x)/std(y), well below the raw 2.0
    assert 0 < std["coefficients"]["ticker:X1"]["coef"] < 1.5


def test_ols_validation(dataset: ReturnDataset) -> None:
    with pytest.raises(ValueError, match="cov must be one of"):
        fit_ols(dataset, "ticker:Y", ["ticker:X1"], cov="bogus")
    with pytest.raises(ValueError, match="ci_level"):
        fit_ols(dataset, "ticker:Y", ["ticker:X1"], ci_level=1.2)
    with pytest.raises(ValueError, match="hac_lags"):
        fit_ols(dataset, "ticker:Y", ["ticker:X1"], cov="HAC", hac_lags=0)


def test_ridge_fixed_alpha(dataset: ReturnDataset) -> None:
    out = fit_ridge(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"], alpha=1e-4)
    assert out["model"] == "ridge"
    assert out["alpha_selection"] == "fixed"
    assert out["alpha"] == pytest.approx(1e-4)
    assert out["cv_folds"] is None
    # nearly-unpenalised ridge stays close to OLS truth (original units)
    assert out["coefficients"]["ticker:X1"] == pytest.approx(TRUE_B1, abs=0.1)
    assert out["standardized_coefficients"] is not None
    assert out["r_squared"] > 0.99
    json.dumps(out)


def test_ridge_cv_selects_alpha_from_grid(dataset: ReturnDataset) -> None:
    out = fit_ridge(dataset, "ticker:Y", ["ticker:X1", "ticker:X2"], cv_folds=5)
    assert out["alpha_selection"] == "cv"
    assert out["cv_folds"] == 5
    assert 1e-4 <= out["alpha"] <= 1e4
    assert out["cv_r_squared"] is not None


def test_lasso_zeroes_irrelevant_regressor(dataset: ReturnDataset) -> None:
    out = fit_lasso(
        dataset, "ticker:Y", ["ticker:X1", "ticker:X2", "ticker:X3"], alpha=0.002
    )
    assert out["model"] == "lasso"
    assert "ticker:X1" in out["selected_regressors"]
    assert "ticker:X3" not in out["selected_regressors"]
    assert out["n_nonzero"] == len(out["selected_regressors"])
    assert out["coefficients"]["ticker:X3"] == 0.0
    json.dumps(out)


def test_lasso_cv_path(dataset: ReturnDataset) -> None:
    out = fit_lasso(dataset, "ticker:Y", ["ticker:X1", "ticker:X2", "ticker:X3"])
    assert out["alpha_selection"] == "cv"
    assert out["alpha"] > 0
    # CV lasso keeps the two real regressors on this near-noiseless fixture
    assert "ticker:X1" in out["selected_regressors"]
    assert "ticker:X2" in out["selected_regressors"]


def test_regularized_validation(dataset: ReturnDataset) -> None:
    with pytest.raises(ValueError, match="alpha must be greater"):
        fit_ridge(dataset, "ticker:Y", ["ticker:X1"], alpha=0.0)
    with pytest.raises(ValueError, match="cv_folds"):
        fit_lasso(dataset, "ticker:Y", ["ticker:X1"], cv_folds=1)


def test_no_series_in_payload(dataset: ReturnDataset) -> None:
    """Coefficients and diagnostics only — never residual/fitted series."""
    for payload in (
        fit_ols(dataset, "ticker:Y", ["ticker:X1"]),
        fit_ridge(dataset, "ticker:Y", ["ticker:X1"], alpha=0.1),
        fit_lasso(dataset, "ticker:Y", ["ticker:X1"], alpha=0.1),
    ):
        flat = json.dumps(payload)
        assert "resid_series" not in flat
        for value in payload.values():
            assert not (isinstance(value, list) and len(value) > 12)
