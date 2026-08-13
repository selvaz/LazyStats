"""The panel fit must not depend on the units the returns are expressed in.

This guards something that was already lost once, in a way nothing detected.

The engine standardizes each column before fitting because hmmlearn's EM is
tuned for roughly unit variance, while daily log returns sit around 1e-4. Losing
that does not raise: the fit still returns a model, the pipeline still writes it,
and the report still renders. What changes is which model gets chosen — in
production, 21 of 90 symbols came out with a different number of states, TLT
among them, and the only visible symptom was a "Model is not converging" warning
in a log nobody reads.

So these tests fit the same series twice, once in fractions and once in percent,
and require the same answer. Scaling the data by a hundred is the same
perturbation standardization removes, and without it the two disagree.
"""
from __future__ import annotations

import pytest

pytest.importorskip("hmmlearn")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lazystats.regimes import MSRegimeEngine  # noqa: E402
from lazystats.regimes.estimation import annualized_states  # noqa: E402


def series(scale: float = 1.0) -> pd.DataFrame:
    """A two-regime daily return series: quiet, then turbulent, then quiet."""
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(0.0004, 0.004, 700),
                             rng.normal(-0.0012, 0.021, 350),
                             rng.normal(0.0004, 0.004, 500)])
    index = pd.bdate_range("2018-01-01", periods=len(values))
    return pd.DataFrame({"X": values * scale}, index=index)


@pytest.fixture(scope="module")
def fits():
    engine = lambda: MSRegimeEngine(S_max=3, n_starts=10, random_state=123)  # noqa: E731
    return engine().fit(series(1.0)), engine().fit(series(100.0))


def test_the_engine_standardizes_panel_fits_by_default(fits):
    """The opt-out exists; the default is what production runs."""
    assert MSRegimeEngine(S_max=3).standardize is True


def test_the_same_series_in_percent_finds_the_same_number_of_states(fits):
    """The shape of the failure that reached production: TLT came out with two
    states in one scale and three in the other.

    Stated for the record rather than as the guard. This series is
    well-conditioned enough that the state count survives losing
    standardization, so removing the feature does not make this fail — the test
    below, on the day-by-day assignments, is the one that bites. Reproducing the
    count flip needs a symbol like TLT and its whole history.
    """
    fractions, percent = fits
    assert int(fractions.meta["X"]["S"]) == int(percent.meta["X"]["S"])


def test_the_regime_assignments_are_the_same_day_by_day(fits):
    fractions, percent = fits
    assert (fractions.panel["X_state"].tolist() == percent.panel["X_state"].tolist())


def test_the_parameters_come_back_in_the_units_they_went_in(fits):
    """De-standardization: means_ and covars_ are the only scale-dependent
    outputs, and they must be returned in the column's own units. A fit whose
    parameters stayed standardized would report every symbol's volatility as
    one."""
    fractions, percent = fits
    thin = annualized_states(fractions.meta["X"]["means_"],
                             fractions.meta["X"]["covars_"],
                             fractions.meta["X"]["labels"])
    wide = annualized_states(percent.meta["X"]["means_"],
                             percent.meta["X"]["covars_"],
                             percent.meta["X"]["labels"])
    for a, b in zip(thin, wide, strict=True):
        assert b["annualized_volatility"] == pytest.approx(
            a["annualized_volatility"] * 100, rel=1e-6)
        assert b["annualized_mean_return"] == pytest.approx(
            a["annualized_mean_return"] * 100, rel=1e-6)


def test_a_daily_scale_volatility_is_not_reported_as_one(fits):
    """The shape a lost de-standardization would take: plausible numbers, all
    of them wrong, and nothing failing."""
    fractions, _ = fits
    vols = [s["annualized_volatility"] for s in annualized_states(
        fractions.meta["X"]["means_"], fractions.meta["X"]["covars_"],
        fractions.meta["X"]["labels"])]
    assert all(0.0 < v < 2.0 for v in vols), vols
    assert max(vols) > min(vols) * 1.5


def test_turning_it_off_is_possible_and_visibly_different(fits):
    """Not a preference: the opt-out has to remain, and the two paths have to be
    distinguishable, or this whole file would pass on a stub."""
    raw = MSRegimeEngine(S_max=3, n_starts=10, random_state=123,
                         standardize=False).fit(series(1.0))
    standardized, _ = fits
    assert raw.meta["X"]["loglik"] != standardized.meta["X"]["loglik"]
