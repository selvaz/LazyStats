# LazyStats

Pure statistical library of the LazyBridge ecosystem. Stdlib-only core —
**zero hard dependencies**, no lazybridge, no LLM wrapping (the LLM tool
surface lives in [LazyTools](https://github.com/selvaz/LazyTools):
`statistical_analysis` wraps `core/`, `RegimeTools` wraps `regimes/`).

```
src/lazystats/
  core/     return_volatility / return_correlation / return_outliers + helpers
  models/   ReturnDataset (shape-compatible with the LazyTools wrapper)
  io/
    datahub.py   default data path: market-data-hub (lazy import)
    depot.py     SQLite result depot with mandatory provenance
    local.py     notebook-only CSV/DataFrame loaders — NEVER LLM tools
  regimes/  HMM / Markov-switching regime engines — install extra:
            lazystats[regimes] (numpy, pandas, matplotlib, hmmlearn,
            scikit-learn; the rest of the library stays dependency-free)
  regression/  OLS / Ridge / Lasso wrappers — install extra:
            lazystats[regression] (numpy, statsmodels, scikit-learn;
            imported lazily, `import lazystats` stays dependency-free)
```

## Regimes

`lazystats.regimes` fits Gaussian/categorical/multivariate HMMs and
Markov-switching models over return series, with automatic state-count
selection, window comparison, regime summaries/changes, plot generation, and
a SQLite depot for fitted results and parameter sets.

Data comes exclusively from market-data-hub (`load_from_datahub`) — the hub
is the ecosystem's sole downloader; nothing in `regimes/` fetches data or
reads arbitrary files on the agent path. Results serialize to the
`lazydatacore` contract (`to_analysis_results`).

```python
from lazystats.regimes import load_from_datahub, fit_regimes, get_regime_summary

load_from_datahub("SPY", start="2020-01-01", frequency="W", data_key="spy_w")
fit = fit_regimes(data_key="spy_w", result_key="spy_fit")
print(get_regime_summary(result_key="spy_fit"))
```

## Usage (core statistics)

```python
from lazystats import return_volatility, standardize
from lazystats.io.datahub import load_returns

ds = load_returns("SPY,TLT", start="2024-01-01", frequency="W")
print(return_volatility(ds, frequency="W")["volatility"])
zscored = standardize(ds)  # callable dataset post-transform (also: demean)
```

## Regression (OLS / Ridge / Lasso)

`lazystats.regression` wraps statsmodels (OLS with robust/HAC standard
errors) and scikit-learn (Ridge/Lasso, cross-validated alpha by default) —
no hand-rolled math. Input is the same `ReturnDataset` panel; output is a
plain JSON-serialisable dict of coefficients and diagnostics, never residual
or fitted series. Univariate regression is simply the 1-regressor case.

```python
from lazystats.io.datahub import load_returns
from lazystats.regression import fit_ols, fit_ridge, fit_lasso

ds = load_returns("SPY,QQQ,TLT", start="2020-01-01", frequency="W")
ols = fit_ols(ds, "ticker:SPY", ["ticker:QQQ", "ticker:TLT"], cov="HAC")
ridge = fit_ridge(ds, "ticker:SPY")          # alpha=None -> RidgeCV
lasso = fit_lasso(ds, "ticker:SPY")          # alpha=None -> LassoCV
print(ols["coefficients"], ridge["alpha"], lasso["selected_regressors"])
```

## Install

LazyStats is distributed from GitHub (only LazyBridge is on PyPI). These
pull the current `main`; append `@vX.Y.Z` to pin a release tag:

```
pip install "lazystats @ git+https://github.com/selvaz/LazyStats.git"
pip install "lazystats[regimes] @ git+https://github.com/selvaz/LazyStats.git"
```
