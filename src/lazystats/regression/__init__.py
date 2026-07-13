"""lazystats.regression — OLS / Ridge / Lasso over ReturnDataset panels.

Optional tier (extra: ``lazystats[regression]``), same contract as
``regimes``: not re-exported by the top-level package, heavy dependencies
(numpy, statsmodels, scikit-learn) imported lazily inside function bodies so
``import lazystats.regression`` succeeds on a bare interpreter.

The math is entirely delegated to statsmodels (OLS + robust covariances) and
scikit-learn (Ridge/Lasso + cross-validated alpha). Results are plain
JSON-serialisable dicts of coefficients and diagnostics — never residual or
fitted series. LLM-facing wrapping (caps, envelopes, hub metadata) lives in
LazyTools, mirroring ``core``.
"""

from lazystats.regression.design import prepare_design
from lazystats.regression.linear import fit_lasso, fit_ols, fit_ridge

__all__ = ["fit_lasso", "fit_ols", "fit_ridge", "prepare_design"]
