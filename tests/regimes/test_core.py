"""
Test suite for markov_hmm — Gaussian HMM regime detection library.

Organised into groups:
  - helpers     : math utility functions (_ensure_2d, BIC/AIC/HQIC, …)
  - reorder     : reorder_fitresult + regime_labels_from_S
  - msgaussian  : MSGaussianHMM fitting and output shapes
  - selection   : select_num_regimes + fit_with_auto_S
  - engine      : MSRegimeEngine → RegimeRun (univariate_panel & multivariate)
  - discrete    : fit_discrete_hmm + fit_with_auto_S_categorical
  - multivar    : fit_multivar_hmm + fit_with_auto_S_multivar

All tests use synthetic data only — no network calls.
Run with:  pytest tests/test_core.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazystats.regimes.core import (
    # helpers
    _ensure_2d,
    _count_free_params,
    _bic,
    _aic,
    _hqic,
    _make_sticky_transmat,
    _expected_durations,
    _state_vol_measure,
    # reorder / labels
    reorder_fitresult,
    regime_labels_from_S,
    # core
    FitResult,
    FitWarning,
    MSGaussianHMM,
    # user-supplied parameters
    HMMParams,
    infer_with_params,
    fit_from_params,
    select_num_regimes,
    fit_with_auto_S,
    # engine
    MSRegimeEngine,
    RegimeRun,
    # discrete
    fit_discrete_hmm,
    fit_with_auto_S_categorical,
    _encode_categorical_obs,
    # multivariate independent-emission
    fit_multivar_hmm,
    fit_with_auto_S_multivar,
)
from tests.regimes.conftest import make_fake_fitresult, make_discrete_1d


# ============================================================================
# Group 1: Helper functions
# ============================================================================

class TestEnsure2D:
    def test_1d_becomes_column_vector(self):
        y = np.arange(10, dtype=float)
        out = _ensure_2d(y)
        assert out.shape == (10, 1)

    def test_2d_array_unchanged(self):
        y = np.arange(20, dtype=float).reshape(10, 2)
        out = _ensure_2d(y)
        assert out.shape == (10, 2)

    def test_list_converted_to_2d(self):
        out = _ensure_2d([1.0, 2.0, 3.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 1)

    def test_3d_raises_value_error(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            _ensure_2d(np.zeros((3, 4, 5)))


class TestCountFreeParams:
    def test_diag_s2_k1(self):
        # start=(S-1)=1, trans=S*(S-1)=2, means=S*k=2, cov=S*k=2 → 7
        assert _count_free_params(2, 1, "diag", False) == 7

    def test_full_s2_k2(self):
        # start=1, trans=2, means=4, cov=S*(k*(k+1)//2)=2*3=6 → 13
        assert _count_free_params(2, 2, "full", False) == 13

    def test_shared_mean_has_fewer_params(self):
        n_free   = _count_free_params(3, 2, "diag", False)
        n_shared = _count_free_params(3, 2, "diag", True)
        assert n_shared < n_free

    def test_invalid_cov_type_raises(self):
        with pytest.raises(ValueError):
            _count_free_params(2, 1, "spherical", False)

    def test_increases_with_S(self):
        n2 = _count_free_params(2, 1, "diag", False)
        n3 = _count_free_params(3, 1, "diag", False)
        assert n3 > n2

    def test_increases_with_k(self):
        n1 = _count_free_params(2, 1, "full", False)
        n2 = _count_free_params(2, 2, "full", False)
        assert n2 > n1


class TestInformationCriteria:
    def test_bic_formula(self):
        # BIC = k*log(n) - 2*loglik
        bic = _bic(-100.0, 5, 200)
        expected = 5 * np.log(200) - 2 * (-100.0)
        assert abs(bic - expected) < 1e-10

    def test_aic_formula(self):
        aic = _aic(-100.0, 5)
        expected = 2 * 5 - 2 * (-100.0)
        assert abs(aic - expected) < 1e-10

    def test_hqic_formula(self):
        hqic = _hqic(-100.0, 5, 200)
        expected = -2 * (-100.0) + 2 * 5 * np.log(np.log(200))
        assert abs(hqic - expected) < 1e-10

    def test_bic_increases_with_more_params(self):
        bic5  = _bic(-100.0, 5, 200)
        bic10 = _bic(-100.0, 10, 200)
        assert bic10 > bic5

    def test_all_criteria_decrease_with_higher_loglik(self):
        for fn in [lambda ll: _bic(ll, 5, 200),
                   lambda ll: _aic(ll, 5),
                   lambda ll: _hqic(ll, 5, 200)]:
            assert fn(-50.0) < fn(-100.0)


class TestStickyTransmat:
    def test_rows_sum_to_one(self):
        rng = np.random.RandomState(0)
        P = _make_sticky_transmat(3, 0.9, rng)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(3), atol=1e-10)

    def test_all_entries_positive(self):
        rng = np.random.RandomState(0)
        P = _make_sticky_transmat(4, 0.95, rng)
        assert np.all(P > 0)

    def test_diagonal_dominant(self):
        rng = np.random.RandomState(0)
        P = _make_sticky_transmat(4, 0.95, rng)
        diag    = np.diag(P)
        off_max = (P - np.diag(diag)).max()
        assert diag.min() > off_max

    def test_s1_is_identity(self):
        rng = np.random.RandomState(0)
        P = _make_sticky_transmat(1, 0.9, rng)
        np.testing.assert_allclose(P, [[1.0]])

    def test_shape_correct(self):
        rng = np.random.RandomState(1)
        for S in (2, 3, 5):
            P = _make_sticky_transmat(S, 0.9, rng)
            assert P.shape == (S, S)


class TestExpectedDurations:
    def test_persistent_state(self):
        P = np.array([[0.99, 0.01], [0.05, 0.95]])
        dur = _expected_durations(P)
        assert abs(dur[0] - 100.0) < 2.0  # 1/(1-0.99)
        assert abs(dur[1] - 20.0) < 1.0   # 1/(1-0.95)

    def test_unstable_state_has_low_duration(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        dur = _expected_durations(P)
        assert np.all(dur == pytest.approx(2.0, abs=0.1))

    def test_floor_prevents_zero_division(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        dur = _expected_durations(P)
        assert np.all(np.isfinite(dur))
        assert np.all(dur >= 1.0)


# ============================================================================
# Group 2: State reordering
# ============================================================================

class TestReorderFitResult:
    def test_vol_ascending_orders_low_to_high(self):
        res = make_fake_fitresult(S=2, cov_type="diag")
        # state 0 starts with HIGH vol (4.0), state 1 with LOW vol (0.1)
        r = reorder_fitresult(res, by="vol", ascending=True)
        vols = _state_vol_measure(r.covars_, r.cov_type)
        assert vols[0] < vols[1], "After ascending reorder, state 0 must have lowest vol"

    def test_vol_descending_orders_high_to_low(self):
        res = make_fake_fitresult(S=2, cov_type="diag")
        r = reorder_fitresult(res, by="vol", ascending=False)
        vols = _state_vol_measure(r.covars_, r.cov_type)
        assert vols[0] > vols[1], "After descending reorder, state 0 must have highest vol"

    def test_mean_ascending(self):
        res = make_fake_fitresult(S=3, cov_type="diag", seed=99)
        r = reorder_fitresult(res, by="mean", ascending=True)
        m = r.means_.mean(axis=1)
        assert m[0] <= m[1] <= m[2]

    def test_s1_returns_same_object(self):
        res = make_fake_fitresult(S=1)
        r = reorder_fitresult(res)
        assert r.S == 1
        assert len(r.warnings) == len(res.warnings)

    def test_transmat_rows_still_sum_to_one(self):
        res = make_fake_fitresult(S=3, cov_type="diag")
        r = reorder_fitresult(res, by="vol", ascending=True)
        np.testing.assert_allclose(r.transmat_.sum(axis=1), np.ones(3), atol=1e-12)

    def test_gamma_rows_still_sum_to_one(self):
        T = 60
        res = make_fake_fitresult(S=3, T=T)
        r = reorder_fitresult(res, by="vol", ascending=True)
        np.testing.assert_allclose(r.gamma_.sum(axis=1), np.ones(T), atol=1e-12)

    def test_viterbi_values_in_range_after_reorder(self):
        S = 4
        res = make_fake_fitresult(S=S, T=80)
        r = reorder_fitresult(res, by="vol", ascending=True)
        assert r.viterbi_path_.min() >= 0
        assert r.viterbi_path_.max() < S

    def test_info_state_reorder_warning_appended(self):
        res = make_fake_fitresult(S=2)
        r = reorder_fitresult(res)
        codes = [w.code for w in r.warnings]
        assert "INFO_STATE_REORDER" in codes

    def test_output_shapes_preserved(self):
        S, T, k = 3, 100, 2
        res = make_fake_fitresult(S=S, T=T, k=k, cov_type="diag")
        r = reorder_fitresult(res, by="vol", ascending=True)
        assert r.startprob_.shape   == (S,)
        assert r.transmat_.shape    == (S, S)
        assert r.means_.shape       == (S, k)
        assert r.covars_.shape      == (S, k)
        assert r.gamma_.shape       == (T, S)
        assert r.viterbi_path_.shape == (T,)

    def test_invalid_by_raises(self):
        res = make_fake_fitresult(S=2)
        with pytest.raises(ValueError):
            reorder_fitresult(res, by="unknown")

    def test_full_cov_type_reorder(self):
        res = make_fake_fitresult(S=2, k=2, cov_type="full")
        r = reorder_fitresult(res, by="vol", ascending=True)
        vols = _state_vol_measure(r.covars_, r.cov_type)
        assert vols[0] <= vols[1]


class TestRegimeLabelsFromS:
    @pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 6, 7])
    def test_label_count_equals_S(self, S):
        assert len(regime_labels_from_S(S)) == S

    def test_s2_canonical_labels(self):
        assert regime_labels_from_S(2) == ["Low Vol", "High Vol"]

    def test_s3_canonical_labels(self):
        assert regime_labels_from_S(3) == ["Low Vol", "Mid Vol", "High Vol"]

    def test_all_labels_are_strings(self):
        for S in range(1, 8):
            labels = regime_labels_from_S(S)
            assert all(isinstance(lb, str) for lb in labels)


# ============================================================================
# Group 3: MSGaussianHMM
# ============================================================================

class TestMSGaussianHMM:
    """Tests for the multi-start wrapper. Uses small n_starts & n_iter for speed."""

    @pytest.fixture(scope="class")
    def fitted(self, Y2_1d):
        ms = MSGaussianHMM(S=2, n_starts=5, cov_type="diag",
                           random_state=42, n_iter=100)
        ms.fit(Y2_1d)
        return ms, Y2_1d

    def test_fit_returns_self(self, Y2_1d):
        ms = MSGaussianHMM(S=2, n_starts=3, cov_type="diag", random_state=0, n_iter=30)
        assert ms.fit(Y2_1d) is ms

    def test_best_result_not_none(self, fitted):
        ms, _ = fitted
        assert ms.best_result_ is not None

    def test_loglik_finite(self, fitted):
        ms, _ = fitted
        assert np.isfinite(ms.best_result_.loglik)

    def test_S_attribute(self, fitted):
        ms, _ = fitted
        assert ms.best_result_.S == 2

    def test_startprob_sums_to_one(self, fitted):
        ms, _ = fitted
        assert abs(ms.best_result_.startprob_.sum() - 1.0) < 1e-10

    def test_transmat_rows_sum_to_one(self, fitted):
        ms, _ = fitted
        P = ms.best_result_.transmat_
        np.testing.assert_allclose(P.sum(axis=1), np.ones(2), atol=1e-10)

    def test_gamma_rows_sum_to_one(self, fitted):
        ms, Y = fitted
        g = ms.best_result_.gamma_
        np.testing.assert_allclose(g.sum(axis=1), np.ones(Y.shape[0]), atol=1e-8)

    def test_gamma_shape(self, fitted):
        ms, Y = fitted
        assert ms.best_result_.gamma_.shape == (Y.shape[0], 2)

    def test_viterbi_shape(self, fitted):
        ms, Y = fitted
        assert ms.best_result_.viterbi_path_.shape == (Y.shape[0],)

    def test_viterbi_values_in_range(self, fitted):
        ms, _ = fitted
        vp = ms.best_result_.viterbi_path_
        assert vp.min() >= 0 and vp.max() <= 1

    def test_means_shape(self, fitted):
        ms, Y = fitted
        assert ms.best_result_.means_.shape == (2, Y.shape[1])

    def test_covars_shape_diag(self, fitted):
        ms, Y = fitted
        k = Y.shape[1]
        shape = ms.best_result_.covars_.shape
        # hmmlearn ≥0.3 stores diag covars as (S, k, k); older as (S, k)
        assert shape[0] == 2 and shape[1] == k

    def test_all_results_sorted_by_loglik(self, fitted):
        ms, _ = fitted
        lls = [r.loglik for r in ms.all_results_]
        assert lls == sorted(lls, reverse=True)

    def test_full_cov_type_shape(self, Y2_1d):
        ms = MSGaussianHMM(S=2, n_starts=3, cov_type="full", random_state=1, n_iter=50)
        ms.fit(Y2_1d)
        # diag covariance for 1 feature stored as (S, 1, 1) by hmmlearn full
        assert ms.best_result_.covars_.shape == (2, 1, 1)

    def test_bic_value_reasonable(self, fitted):
        ms, _ = fitted
        assert np.isfinite(ms.best_result_.bic)
        assert ms.best_result_.bic > 0

    def test_converged_attribute_is_bool(self, fitted):
        ms, _ = fitted
        assert isinstance(ms.best_result_.converged, bool)


# ============================================================================
# Group 3b: HMMParams / infer_with_params / fit_from_params
# ============================================================================

class TestHMMParams:
    @pytest.fixture
    def params(self):
        return HMMParams(
            startprob_=[0.5, 0.5],
            transmat_=[[0.9, 0.1], [0.2, 0.8]],
            means_=[[0.0], [0.5]],
            covars_=[0.25, 4.0],          # variances, diag, k=1
            cov_type="diag",
        )

    def test_S_and_features(self, params):
        assert params.S == 2 and params.n_features == 1

    def test_startprob_normalized(self):
        p = HMMParams(startprob_=[1, 3], transmat_=[[2, 2], [1, 1]],
                      means_=[[0], [1]], covars_=[1, 1])
        assert abs(p.startprob_.sum() - 1.0) < 1e-12
        np.testing.assert_allclose(p.transmat_.sum(axis=1), [1.0, 1.0])

    def test_dict_roundtrip(self, params):
        p2 = HMMParams.from_dict(params.to_dict())
        np.testing.assert_allclose(p2.transmat_, params.transmat_)
        np.testing.assert_allclose(p2.covars_, params.covars_)

    def test_diag_from_3d_covars_extracts_diagonal(self):
        cov = np.array([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]])
        p = HMMParams(startprob_=[.5, .5], transmat_=[[.9, .1], [.1, .9]],
                      means_=[[0, 0], [1, 1]], covars_=cov, cov_type="diag")
        assert p.covars_.shape == (2, 2)
        np.testing.assert_allclose(p.covars_, [[1, 2], [3, 4]])

    def test_bad_transmat_shape_raises(self):
        with pytest.raises(ValueError):
            HMMParams(startprob_=[.5, .5], transmat_=[[1.0]],
                      means_=[[0], [1]], covars_=[1, 1])

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError):
            HMMParams(startprob_=[.5, .5], transmat_=[[.9, .1], [.1, .9]],
                      means_=[[0], [1]], covars_=[-1, 1])

    def test_non_stochastic_without_renormalize_raises(self):
        with pytest.raises(ValueError):
            HMMParams(startprob_=[.5, .4], transmat_=[[.9, .1], [.1, .9]],
                      means_=[[0], [1]], covars_=[1, 1], renormalize=False)


class TestInferAndFitFromParams:
    @pytest.fixture
    def params(self):
        return HMMParams(
            startprob_=[0.5, 0.5],
            transmat_=[[0.9, 0.1], [0.2, 0.8]],
            means_=[[0.0], [0.5]],
            covars_=[0.25, 4.0],
            cov_type="diag",
        )

    def test_infer_holds_params_fixed(self, params, Y2_1d):
        r = infer_with_params(Y2_1d, params)
        assert r.n_iter == 0 and r.converged is True
        np.testing.assert_allclose(r.transmat_, params.transmat_)
        np.testing.assert_allclose(r.means_, params.means_)

    def test_infer_output_shapes(self, params, Y2_1d):
        r = infer_with_params(Y2_1d, params)
        assert r.gamma_.shape == (Y2_1d.shape[0], 2)
        assert r.viterbi_path_.shape == (Y2_1d.shape[0],)
        assert np.isfinite(r.loglik)

    def test_feature_mismatch_raises(self, params):
        with pytest.raises(ValueError):
            infer_with_params(np.zeros((10, 3)), params)

    def test_warm_start_does_not_worsen_loglik(self, params, Y2_1d):
        r_inf = infer_with_params(Y2_1d, params)
        r_fit = fit_from_params(Y2_1d, params, n_iter=200)
        assert r_fit.loglik >= r_inf.loglik - 1e-6

    def test_empty_train_equals_inference(self, params, Y2_1d):
        r_inf = infer_with_params(Y2_1d, params)
        r_eq = fit_from_params(Y2_1d, params, train_params="", n_iter=0)
        assert abs(r_eq.loglik - r_inf.loglik) < 1e-6

    def test_frozen_means_unchanged(self, params, Y2_1d):
        r = fit_from_params(Y2_1d, params, n_iter=100, train_params="tc")
        np.testing.assert_allclose(r.means_, params.means_)

    def test_fitresult_param_roundtrip(self, Y2_1d):
        ms = MSGaussianHMM(S=2, n_starts=4, random_state=0, n_iter=80).fit(Y2_1d)
        p = HMMParams.from_fitresult(ms.best_result_)
        r = infer_with_params(Y2_1d, p)
        assert abs(r.loglik - ms.best_result_.loglik) < 1e-6


class TestDiversifyStarts:
    def test_diversify_increases_distinct_starts(self, Y2_1d):
        off = MSGaussianHMM(S=3, n_starts=10, random_state=42,
                            diversify_starts=False, n_iter=120).fit(Y2_1d)
        on = MSGaussianHMM(S=3, n_starts=10, random_state=42,
                           diversify_starts=True, n_iter=120).fit(Y2_1d)
        n_off = len({round(r.loglik, 4) for r in off.all_results_})
        n_on = len({round(r.loglik, 4) for r in on.all_results_})
        assert n_on >= n_off


# ============================================================================
# Group 4: State selection
# ============================================================================

class TestSelectNumRegimes:
    @pytest.fixture(scope="class")
    def sel(self, Y2_1d):
        return select_num_regimes(
            Y2_1d, S_max=4, n_starts=5, criterion="bic",
            random_state=42, n_iter=80,
        )

    def test_best_S_is_int_in_range(self, sel):
        assert isinstance(sel["best_S"], int)
        assert 1 <= sel["best_S"] <= 4

    def test_per_S_has_all_entries(self, sel):
        assert set(sel["per_S"].keys()) == {1, 2, 3, 4}

    def test_each_per_S_has_score(self, sel):
        for S, v in sel["per_S"].items():
            assert "score" in v

    def test_each_per_S_has_rejected_bool(self, sel):
        for S, v in sel["per_S"].items():
            assert isinstance(v["rejected"], bool)

    def test_criterion_stored_in_output(self, sel):
        assert sel["criterion"] == "bic"

    def test_best_S_not_none_even_if_all_rejected(self):
        # Very short series — some S values will be rejected, best_S still returned
        rng = np.random.RandomState(7)
        Y = rng.randn(30, 1)
        sel = select_num_regimes(Y, S_max=3, n_starts=3, random_state=0, n_iter=30)
        assert sel["best_S"] is not None

    def test_aic_criterion(self, Y2_1d):
        sel = select_num_regimes(Y2_1d, S_max=3, n_starts=3, criterion="aic",
                                  random_state=0, n_iter=50)
        assert sel["criterion"] == "aic"
        assert 1 <= sel["best_S"] <= 3

    def test_hqic_criterion(self, Y2_1d):
        sel = select_num_regimes(Y2_1d, S_max=3, n_starts=3, criterion="hqic",
                                  random_state=0, n_iter=50)
        assert sel["criterion"] == "hqic"


class TestFitWithAutoS:
    @pytest.fixture(scope="class")
    def out(self, Y2_1d):
        return fit_with_auto_S(
            Y2_1d, S_max=4, n_starts=5, criterion="bic",
            random_state=42, n_iter=80,
        )

    def test_best_S_returned(self, out):
        assert isinstance(out["best_S"], int)
        assert 1 <= out["best_S"] <= 4

    def test_final_result_present(self, out):
        assert out["final_result"] is not None

    def test_final_result_loglik_finite(self, out):
        assert np.isfinite(out["final_result"].loglik)

    def test_final_model_present(self, out):
        assert out["final_model"] is not None

    def test_selection_dict_present(self, out):
        assert "selection" in out
        assert "per_S" in out["selection"]

    def test_criterion_stored(self, out):
        assert out["criterion"] == "bic"

    def test_2state_data_selects_2_or_3(self, Y2_1d):
        out = fit_with_auto_S(
            Y2_1d, S_max=4, n_starts=10, criterion="bic",
            random_state=42, n_iter=100,
        )
        # BIC is conservative — allow S=2 or S=3
        assert out["best_S"] in {2, 3}


# ============================================================================
# Group 5: MSRegimeEngine / RegimeRun
# ============================================================================

class TestMSRegimeEngineUnivariate:
    @pytest.fixture(scope="class")
    def run(self, multivar_df):
        engine = MSRegimeEngine(S_max=3, n_starts=5, random_state=42)
        return engine.fit(multivar_df, model="panel")

    def test_returns_regime_run_instance(self, run):
        assert isinstance(run, RegimeRun)

    def test_panel_length_matches_input(self, run, multivar_df):
        assert len(run.panel) == len(multivar_df)

    def test_panel_has_value_columns(self, run):
        for col in ["X0", "X1"]:
            assert f"{col}_value" in run.panel.columns

    def test_panel_has_state_columns(self, run):
        for col in ["X0", "X1"]:
            assert f"{col}_state" in run.panel.columns

    def test_panel_has_highvol_columns(self, run):
        for col in ["X0", "X1"]:
            assert f"{col}_highvol" in run.panel.columns

    def test_panel_has_prob_hv_columns(self, run):
        for col in ["X0", "X1"]:
            assert f"P_{col}_HV" in run.panel.columns

    def test_state_values_non_negative(self, run):
        for col in ["X0", "X1"]:
            assert run.panel[f"{col}_state"].min() >= 0

    def test_highvol_is_binary(self, run):
        for col in ["X0", "X1"]:
            vals = set(run.panel[f"{col}_highvol"].unique())
            assert vals.issubset({0, 1})

    def test_prob_hv_in_unit_interval(self, run):
        for col in ["X0", "X1"]:
            p = run.panel[f"P_{col}_HV"]
            assert (p >= 0).all() and (p <= 1).all()

    def test_labels_map_has_all_columns(self, run):
        for col in ["X0", "X1"]:
            assert col in run.labels_map

    def test_meta_has_all_columns(self, run):
        for col in ["X0", "X1"]:
            assert col in run.meta

    def test_rows_attribute_correct(self, run):
        assert run.rows == ["X0", "X1"]

    def test_meta_model_is_panel(self, run):
        assert run.meta.get("_MODEL_") == "panel"

    def test_set_theme_bloomberg(self, run):
        run.set_theme("bloomberg")  # should not raise

    def test_set_theme_light(self, run):
        run.set_theme("light")

    def test_set_theme_minimal(self, run):
        run.set_theme("minimal")

    def test_set_color_alias(self, run):
        run.set_color("bloomberg")  # alias for set_theme

    def test_set_theme_invalid_raises(self, run):
        with pytest.raises(ValueError, match="not available"):
            run.set_theme("nonexistent_theme_xyz")

    def test_available_themes_returns_sorted_list(self, run):
        themes = run.available_themes()
        assert isinstance(themes, list)
        assert "bloomberg" in themes
        assert "light" in themes
        assert "minimal" in themes
        assert themes == sorted(themes)

    def test_meta_s_is_positive_int(self, run):
        for col in ["X0", "X1"]:
            assert isinstance(run.meta[col]["S"], int)
            assert run.meta[col]["S"] >= 1

    def test_meta_bic_finite(self, run):
        for col in ["X0", "X1"]:
            assert np.isfinite(run.meta[col]["bic"])


class TestMSRegimeEngineMultivariate:
    @pytest.fixture(scope="class")
    def run(self, multivar_df):
        engine = MSRegimeEngine(S_max=3, n_starts=5, random_state=42)
        return engine.fit(multivar_df, model="joint_full")

    def test_returns_regime_run_instance(self, run):
        assert isinstance(run, RegimeRun)

    def test_all_series_share_same_state(self, run):
        # In multivariate mode, latent state is common to all series
        s0 = run.panel["X0_state"].values
        s1 = run.panel["X1_state"].values
        np.testing.assert_array_equal(s0, s1)

    def test_meta_model_is_joint_full(self, run):
        assert run.meta.get("_MODEL_") == "joint_full"

    def test_multivariate_shared_state_flag(self, run):
        for col in ["X0", "X1"]:
            assert run.meta[col].get("shared_multivariate_state") is True


class TestRegimeRunEdgeCases:
    def test_single_column_dataframe(self):
        import pandas as pd
        rng = np.random.RandomState(5)
        df = pd.DataFrame({"A": rng.randn(150)},
                           index=pd.date_range("2020-01-01", periods=150, freq="W"))
        engine = MSRegimeEngine(S_max=2, n_starts=4, random_state=0)
        run = engine.fit(df, model="panel")
        assert isinstance(run, RegimeRun)
        assert "A_state" in run.panel.columns

    def test_dropna_any_mode(self, multivar_df):
        df_nan = multivar_df.copy()
        df_nan.iloc[5, 0] = np.nan
        engine = MSRegimeEngine(S_max=2, n_starts=3, random_state=0)
        run = engine.fit(df_nan, model="panel", dropna="any")
        assert isinstance(run, RegimeRun)

    def test_panel_posterior_prob_cols_sum_to_one(self):
        rng = np.random.RandomState(7)
        df = pd.DataFrame({"Y": rng.randn(120)},
                           index=pd.date_range("2015-01-01", periods=120, freq="W"))
        engine = MSRegimeEngine(S_max=2, n_starts=3, random_state=0)
        run = engine.fit(df, model="panel")
        S = run.meta["Y"]["S"]
        prob_cols = [f"P_Y_S{s}" for s in range(S)]
        row_sums = run.panel[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, np.ones(len(run.panel)), atol=1e-8)


# ============================================================================
# Group 6: Discrete / Categorical HMM
# ============================================================================

class TestEncodeCategoricalObs:
    def test_1d_encodes_to_range_0_K(self):
        y = np.array([2, 0, 1, 2, 0])
        y_enc, K, sym_map, sym_inv = _encode_categorical_obs(y)
        assert K == 3
        assert y_enc.min() == 0
        assert y_enc.max() == 2

    def test_1d_roundtrip_via_symbol_map(self):
        y = np.array([3, 1, 2, 1, 3, 0])
        y_enc, K, sym_map, sym_inv = _encode_categorical_obs(y)
        decoded = np.array([sym_map[c][0] for c in y_enc])
        np.testing.assert_array_equal(decoded, y)

    def test_2d_encoding_output_shape(self):
        Y = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        y_enc, K, sym_map, sym_inv = _encode_categorical_obs(Y)
        assert y_enc.shape == (4,)
        assert K <= 4  # at most 4 unique rows

    def test_2d_sym_map_has_K_entries(self):
        Y = np.random.RandomState(0).randint(0, 3, size=(50, 2))
        y_enc, K, sym_map, sym_inv = _encode_categorical_obs(Y)
        assert len(sym_map) == K

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            _encode_categorical_obs(np.zeros((3, 3, 3)))


class TestFitDiscreteHMM:
    @pytest.fixture(scope="class")
    def fitted(self, discrete_1d):
        return fit_discrete_hmm(discrete_1d, S=2, n_iter=100, random_state=42)

    def test_s_attribute(self, fitted):
        assert fitted.S == 2

    def test_k_attribute(self, fitted):
        assert fitted.K == 3  # discrete_1d uses K=3

    def test_loglik_finite(self, fitted):
        assert np.isfinite(fitted.loglik)

    def test_emission_shape(self, fitted):
        assert fitted.emissionprob_.shape == (2, 3)

    def test_emission_rows_sum_to_one(self, fitted):
        np.testing.assert_allclose(fitted.emissionprob_.sum(axis=1), np.ones(2), atol=1e-8)

    def test_transmat_rows_sum_to_one(self, fitted):
        np.testing.assert_allclose(fitted.transmat_.sum(axis=1), np.ones(2), atol=1e-8)

    def test_startprob_sums_to_one(self, fitted):
        assert abs(fitted.startprob_.sum() - 1.0) < 1e-8

    def test_gamma_shape(self, fitted, discrete_1d):
        assert fitted.gamma_.shape == (len(discrete_1d), 2)

    def test_gamma_rows_sum_to_one(self, fitted, discrete_1d):
        T = len(discrete_1d)
        np.testing.assert_allclose(fitted.gamma_.sum(axis=1), np.ones(T), atol=1e-8)

    def test_viterbi_shape(self, fitted, discrete_1d):
        assert fitted.viterbi_path_.shape == (len(discrete_1d),)

    def test_viterbi_values_in_valid_range(self, fitted):
        assert fitted.viterbi_path_.min() >= 0
        assert fitted.viterbi_path_.max() <= 1

    def test_bic_positive(self, fitted):
        assert fitted.bic > 0

    def test_2d_input_accepted(self, discrete_2d):
        res = fit_discrete_hmm(discrete_2d, S=2, n_iter=80, random_state=0)
        assert res.S == 2
        assert res.viterbi_path_.shape == (len(discrete_2d),)

    def test_sticky_parameter(self, discrete_1d):
        # sticky=0.8 should not raise and should produce valid output
        res = fit_discrete_hmm(discrete_1d, S=2, n_iter=50, sticky=0.8, random_state=0)
        assert np.isfinite(res.loglik)

    def test_progress_callback_called(self, discrete_1d):
        calls = []
        fit_discrete_hmm(discrete_1d, S=2, n_iter=30, random_state=0,
                          progress_callback=lambda i, ll, c: calls.append(i))
        assert len(calls) > 0


class TestFitWithAutoSCategorical:
    @pytest.fixture(scope="class")
    def out(self, discrete_1d):
        return fit_with_auto_S_categorical(
            discrete_1d, S_max=3, n_starts=3, n_iter=80, random_state=42,
        )

    def test_best_S_is_int_in_range(self, out):
        assert isinstance(out["best_S"], int)
        assert 1 <= out["best_S"] <= 3

    def test_final_result_not_none(self, out):
        assert out["final_result"] is not None

    def test_all_results_count(self, out):
        assert len(out["all_results"]) == 3

    def test_each_result_entry_has_required_keys(self, out):
        for r in out["all_results"]:
            assert "S" in r and "best_bic" in r and "best_loglik" in r

    def test_best_s_gives_lowest_bic(self, out):
        bic_at_best = out["final_result"].bic
        for r in out["all_results"]:
            # Allow numerical tolerance
            assert r["best_bic"] >= bic_at_best - 1e-6

    def test_unsupported_criterion_raises(self, discrete_1d):
        with pytest.raises(ValueError, match="BIC"):
            fit_with_auto_S_categorical(discrete_1d, S_max=2, criterion="aic")


# ============================================================================
# Group 7: Multivariate independent-emission HMM
# ============================================================================

class TestFitMultivarHMM:
    @pytest.fixture(scope="class")
    def fitted(self, discrete_2d):
        return fit_multivar_hmm(discrete_2d, S=2, n_iter=100, random_state=42)

    def test_s_attribute(self, fitted):
        assert fitted.S == 2

    def test_d_attribute(self, fitted, discrete_2d):
        assert fitted.D == discrete_2d.shape[1]

    def test_loglik_finite(self, fitted):
        assert np.isfinite(fitted.loglik)

    def test_gamma_shape(self, fitted, discrete_2d):
        assert fitted.gamma_.shape == (len(discrete_2d), 2)

    def test_gamma_rows_sum_to_one(self, fitted, discrete_2d):
        T = len(discrete_2d)
        np.testing.assert_allclose(fitted.gamma_.sum(axis=1), np.ones(T), atol=1e-8)

    def test_theta_non_negative(self, fitted):
        assert np.all(fitted.emission_theta_ >= 0)

    def test_transmat_rows_sum_to_one(self, fitted):
        np.testing.assert_allclose(fitted.transmat_.sum(axis=1), np.ones(2), atol=1e-8)

    def test_startprob_sums_to_one(self, fitted):
        assert abs(fitted.startprob_.sum() - 1.0) < 1e-8

    def test_viterbi_shape(self, fitted, discrete_2d):
        assert fitted.viterbi_path_.shape == (len(discrete_2d),)

    def test_viterbi_values_valid(self, fitted):
        assert fitted.viterbi_path_.min() >= 0
        assert fitted.viterbi_path_.max() <= 1

    def test_bic_positive(self, fitted):
        assert fitted.bic > 0

    def test_C_attribute_shape(self, fitted, discrete_2d):
        assert fitted.C.shape == (discrete_2d.shape[1],)

    def test_1d_input_reshaped_to_2d(self):
        y = make_discrete_1d(T=120, K=3)
        res = fit_multivar_hmm(y, S=2, n_iter=50, random_state=0)
        assert res.D == 1
        assert res.S == 2

    def test_sticky_accepted(self, discrete_2d):
        res = fit_multivar_hmm(discrete_2d, S=2, n_iter=40, sticky=0.7, random_state=0)
        assert np.isfinite(res.loglik)


class TestFitWithAutoSMultivar:
    @pytest.fixture(scope="class")
    def out(self, discrete_2d):
        return fit_with_auto_S_multivar(
            discrete_2d, S_max=3, n_starts=3, n_iter=80, random_state=42,
        )

    def test_best_S_is_int_in_range(self, out):
        assert isinstance(out["best_S"], int)
        assert 1 <= out["best_S"] <= 3

    def test_final_result_not_none(self, out):
        assert out["final_result"] is not None

    def test_all_results_count(self, out):
        assert len(out["all_results"]) == 3

    def test_each_result_has_required_keys(self, out):
        for r in out["all_results"]:
            assert "S" in r and "best_bic" in r and "best_loglik" in r

    def test_unsupported_criterion_raises(self, discrete_2d):
        with pytest.raises(ValueError, match="BIC"):
            fit_with_auto_S_multivar(discrete_2d, S_max=2, criterion="hqic")


# ============================================================================
# Group 8: FitResult / FitWarning dataclasses
# ============================================================================

class TestFitWarning:
    def test_fields_accessible(self):
        w = FitWarning(code="TEST", severity="info", message="test msg", context={"k": 1})
        assert w.code == "TEST"
        assert w.severity == "info"
        assert w.context["k"] == 1


class TestFitResultSummary:
    def test_summary_returns_string(self):
        res = make_fake_fitresult(S=2, T=50)
        summary = res.summary()
        assert isinstance(summary, str)
        assert "FitResult" in summary

    def test_summary_contains_loglik(self):
        res = make_fake_fitresult(S=3, T=50)
        assert "-500.00" in res.summary()

    def test_summary_empty_gamma(self):
        res = FitResult(
            S=2, cov_type="diag", shared_mean=False, seed=0,
            converged=True, n_iter=5, loglik=-1.0, bic=10.0,
            startprob_=np.array([0.5, 0.5]),
            transmat_=np.array([[0.9, 0.1], [0.1, 0.9]]),
            means_=np.zeros((2, 1)),
            covars_=np.ones((2, 1)),
            gamma_=np.empty((0, 2)),
            viterbi_path_=np.empty(0, dtype=int),
        )
        summary = res.summary()
        assert "n/a" in summary
