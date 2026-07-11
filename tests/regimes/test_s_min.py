# -*- coding: utf-8 -*-
"""Tests for the configurable lower bound ``S_min`` on the regime model-selection API.

Covers the core selection functions, the MSRegimeEngine, the categorical and
tool layers, the provenance record, and the shared ``_validate_S_bounds`` helper.
Fits are kept deliberately small (low T, few starts, small S_max) for speed.
"""
from __future__ import annotations

import pytest

import lazystats.regimes as lz
from lazystats.regimes.core import (
    select_num_regimes,
    MSRegimeEngine,
    _validate_S_bounds,
)

from tests.regimes.conftest import make_2state_1d, make_multivariate_df, make_discrete_1d


# ---------------------------------------------------------------------------
# core.select_num_regimes
# ---------------------------------------------------------------------------
class TestSelectNumRegimes:
    def test_s_min_restricts_scan(self):
        Y = make_2state_1d(T=200, seed=0)
        res = select_num_regimes(Y, S_min=2, S_max=3, n_starts=4, n_iter=80,
                                 random_state=1)
        assert set(res["per_S"].keys()) == {2, 3}
        assert res["best_S"] >= 2

    def test_default_s_min_is_one(self):
        Y = make_2state_1d(T=200, seed=0)
        res = select_num_regimes(Y, S_max=3, n_starts=4, n_iter=80, random_state=1)
        # Backward compatible: scan starts at 1.
        assert set(res["per_S"].keys()) == {1, 2, 3}


# ---------------------------------------------------------------------------
# MSRegimeEngine
# ---------------------------------------------------------------------------
class TestEngineSMin:
    def test_panel_fit_respects_s_min(self):
        df = make_multivariate_df(T=180, k=2, seed=10)
        engine = MSRegimeEngine(S_min=2, S_max=3, n_starts=4, n_iter=80,
                                random_state=3)
        run = engine.fit(df, model="panel")
        for col in df.columns:
            assert run.meta[col]["S"] >= 2

    def test_engine_stores_s_min(self):
        engine = MSRegimeEngine(S_min=2, S_max=4)
        assert engine.S_min == 2
        assert engine.S_max == 4


# ---------------------------------------------------------------------------
# tools.scan_state_counts
# ---------------------------------------------------------------------------
class TestScanStateCounts:
    def test_scan_returns_only_s_min_to_s_max(self):
        Y = make_2state_1d(T=180, seed=0)
        out = lz.scan_state_counts(
            data=Y.tolist(), series_names=["A"],
            model="panel", S_min=2, S_max=4, n_starts=4,
        )
        scores = out["series"]["A"]["scores"]
        s_values = {row["S"] for row in scores}
        assert s_values == {2, 3, 4}
        assert 1 not in s_values


# ---------------------------------------------------------------------------
# tools.fit_regimes + provenance
# ---------------------------------------------------------------------------
class TestFitRegimesToolSMin:
    def test_fit_regimes_s_min_and_provenance(self):
        Y = make_2state_1d(T=200, seed=0)
        out = lz.fit_regimes(
            data=Y.tolist(), series_names=["A"], result_key="smin_r1",
            model="panel", S_min=2, S_max=3, n_starts=4, random_state=1,
        )
        assert out["series"]["A"]["S"] >= 2
        rec = lz.regime_params_load("smin_r1::params")
        assert rec["provenance"]["S_min"] == 2
        assert rec["provenance"]["S_max"] == 3


# ---------------------------------------------------------------------------
# Positional-argument backward compatibility (S_min appended at the end)
# ---------------------------------------------------------------------------
class TestPositionalCompat:
    """S_min is the LAST parameter of the public tool functions, so existing
    positional calls that pass `criterion` (and later options) by position keep
    their meaning instead of binding to S_min."""

    def test_fit_regimes_positional_criterion_binds_correctly(self):
        Y = make_2state_1d(T=160, seed=0)
        # Positional: data, series_names, data_key, result_key, model, S_max, criterion, n_starts
        out = lz.fit_regimes(Y.tolist(), ["A"], "", "pos_r1", "panel", 3, "aic", 4)
        # 'aic' bound to criterion (not S_min); default S_min=1 means S can be 1..3.
        assert 1 <= out["series"]["A"]["S"] <= 3
        rec = lz.regime_params_load("pos_r1::params")
        assert rec["provenance"]["criterion"] == "aic"
        assert rec["provenance"]["S_min"] == 1

    def test_scan_state_counts_positional_criterion(self):
        Y = make_2state_1d(T=160, seed=0)
        # Positional: data, series_names, model, S_max, criterion
        out = lz.scan_state_counts(Y.tolist(), ["A"], "panel", 3, "aic")
        assert out["criterion"] == "aic"
        s_values = {row["S"] for row in out["series"]["A"]["scores"]}
        assert s_values == {1, 2, 3}  # default S_min=1 preserved


# ---------------------------------------------------------------------------
# tools.fit_categorical_regimes
# ---------------------------------------------------------------------------
class TestCategoricalSMin:
    def test_categorical_s_min(self):
        obs = make_discrete_1d(T=180, K=3, seed=3)
        out = lz.fit_categorical_regimes(
            observations=obs.tolist(), S_min=2, S_max=4, n_starts=4, n_iter=60,
            random_state=0,
        )
        assert out["S"] >= 2
        s_values = {row["S"] for row in out["scores"]}
        assert s_values == {2, 3, 4}


# ---------------------------------------------------------------------------
# _validate_S_bounds
# ---------------------------------------------------------------------------
class TestValidation:
    def test_s_min_zero_raises(self):
        with pytest.raises(ValueError):
            _validate_S_bounds(0, 3)

    def test_s_min_greater_than_s_max_raises(self):
        with pytest.raises(ValueError):
            _validate_S_bounds(5, 3)

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            _validate_S_bounds(2.5, 3)
        with pytest.raises(ValueError):
            _validate_S_bounds(2, "3")

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            _validate_S_bounds(True, 3)

    def test_valid_bounds_ok(self):
        # Should not raise.
        _validate_S_bounds(1, 3)
        _validate_S_bounds(2, 2)

    def test_select_num_regimes_validates(self):
        Y = make_2state_1d(T=100, seed=0)
        with pytest.raises(ValueError):
            select_num_regimes(Y, S_min=0, S_max=3, n_starts=2)
        with pytest.raises(ValueError):
            select_num_regimes(Y, S_min=5, S_max=3, n_starts=2)
