"""A BIL-like single-state model has no turbulent alternative."""
import numpy as np
import pandas as pd
import pytest

from lazystats.regimes.core import MSRegimeEngine
from lazystats.regimes.tools import _series_to_dict, get_current_regime
from tests.regimes.conftest import make_fake_fitresult


def test_single_state_tool_readings_are_never_high_vol():
    res = make_fake_fitresult(S=1, T=20)
    series = _series_to_dict("BIL", res, np.zeros(20))
    assert series["high_vol_flag"] == [0] * 20
    assert series["prob_high_vol"] == [0.0] * 20
    np.testing.assert_allclose(series["state_probs"], [[1.0]] * 20)


def test_single_state_current_regime_even_for_legacy_record():
    series = _series_to_dict("BIL", make_fake_fitresult(S=1, T=20), np.zeros(20))
    series["prob_high_vol"] = [1.0] * 20  # record saved before the fix
    out = get_current_regime(fit_result={"series": {"BIL": series}}, series_name="BIL")
    assert out["is_high_vol"] is False
    assert out["prob_high_vol"] == 0.0
    assert out["transition_to_high_vol"] == 0.0


@pytest.mark.parametrize("mode", ["panel", "joint_diag", "joint_full"])
def test_single_state_engine_panel_is_never_high_vol(mode):
    values = np.random.default_rng(123).normal(0.0001, 0.00001, 40)
    frame = pd.DataFrame({"BIL": values}, index=pd.bdate_range("2026-08-01", periods=40))
    run = MSRegimeEngine(S_max=1, n_starts=1, n_iter=10).fit(frame, model=mode)
    assert not run.panel["BIL_highvol"].any()
    assert not run.panel["P_BIL_HV"].any()
