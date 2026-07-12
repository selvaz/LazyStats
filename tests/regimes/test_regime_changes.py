# -*- coding: utf-8 -*-
"""Tests for get_regime_changes + the last-change fields on get_current_regime."""
from __future__ import annotations

import numpy as np
import pytest

from lazystats.regimes import fit_regimes, get_current_regime, get_regime_changes
from lazystats.regimes.tools import _swrite


def _fit_result(states, index=None):
    fr = {"model": "panel", "series": {"SPY": {
        "states": states, "labels": ["Low Vol", "High Vol"],
    }}}
    if index is not None:
        fr["index"] = index
    return fr


# --------------------------------------------------------------------------- #
# get_regime_changes — pure logic via fit_result (no store / no hmmlearn)      #
# --------------------------------------------------------------------------- #
def test_detects_changes_with_dates() -> None:
    idx = ["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24", "2020-01-31", "2020-02-07"]
    out = get_regime_changes(fit_result=_fit_result([0, 0, 1, 1, 1, 0], idx), series_name="SPY")
    assert out["n_changes"] == 2
    assert [c["date"] for c in out["changes"]] == ["2020-01-17", "2020-02-07"]
    assert out["changes"][0]["from_label"] == "Low Vol"
    assert out["changes"][0]["to_label"] == "High Vol"
    assert out["last_change_date"] == "2020-02-07"
    assert out["steps_in_current_regime"] == 1          # only the last bar
    assert out["current_label"] == "Low Vol"


def test_no_change_returns_first_date_and_full_run() -> None:
    idx = ["2020-01-03", "2020-01-10", "2020-01-17"]
    out = get_regime_changes(fit_result=_fit_result([0, 0, 0], idx), series_name="SPY")
    assert out["n_changes"] == 0
    assert out["last_change_date"] == "2020-01-03"
    assert out["steps_in_current_regime"] == 3


def test_integer_position_fallback_when_no_index() -> None:
    out = get_regime_changes(fit_result=_fit_result([0, 1, 1]), series_name="SPY")
    assert out["last_change_date"] == "1"               # positions as strings
    assert out["n_changes"] == 1


def test_last_n_limits_changes_only() -> None:
    idx = [str(i) for i in range(6)]
    out = get_regime_changes(fit_result=_fit_result([0, 1, 0, 1, 0, 1], idx),
                             series_name="SPY", last_n=2)
    assert out["n_changes"] == 5                         # summary over full history
    assert len(out["changes"]) == 2                      # but only last 2 returned


def test_unknown_series_raises() -> None:
    with pytest.raises(KeyError):
        get_regime_changes(fit_result=_fit_result([0, 1]), series_name="NOPE")


# --------------------------------------------------------------------------- #
# end-to-end: fit_regimes stores the index; the query tools read real dates    #
# --------------------------------------------------------------------------- #
def test_fit_regimes_propagates_index_to_changes() -> None:
    rng = np.random.RandomState(0)
    T = 60
    y = np.concatenate([rng.normal(0, 1, 30), rng.normal(0, 5, 30)]).reshape(-1, 1)
    dates = [f"2020-W{i:02d}" for i in range(T)]
    _swrite("data_rc", {"Y": y, "columns": ["SPY"], "index": dates})

    fit_regimes(data_key="data_rc", result_key="res_rc", model="panel",
                S_max=2, n_starts=3, random_state=1)

    chg = get_regime_changes(result_key="res_rc", series_name="SPY")
    # dates come from the stored index, not integer positions
    assert chg["last_change_date"] in dates
    assert chg["steps_in_current_regime"] >= 1

    cur = get_current_regime(result_key="res_rc", series_name="SPY")
    assert cur["last_change_date"] in dates
    assert "steps_in_current_regime" in cur and "n_changes" in cur


def test_depot_roundtrips_index(tmp_path) -> None:
    """Persistent SQLite depot must round-trip the date index (Codex P2)."""
    from lazystats.regimes.db import RegimeDB

    db = RegimeDB(str(tmp_path / "depot.sqlite"))
    full = {
        "model": "panel", "criterion": "bic", "n_timesteps": 4,
        "series": {"SPY": {
            "S": 2, "labels": ["Low Vol", "High Vol"],
            "states": [0, 0, 1, 1], "state_probs": [[1.0, 0.0]] * 4,
            "high_vol_flag": [0, 0, 1, 1], "prob_high_vol": [0.0, 0.0, 1.0, 1.0],
            "regime_stats": [{}, {}], "transition_matrix": [[0.9, 0.1], [0.1, 0.9]],
            "bic": -1.0, "loglik": 1.0,
        }},
        "index": ["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"],
    }
    db.write_result("r", full)
    got = db.read_result("r")
    assert got["index"] == full["index"]            # dates, not "0","1",...

    out = get_regime_changes(fit_result=got, series_name="SPY")
    assert out["last_change_date"] == "2020-01-17"
    assert out["n_changes"] == 1


# ---------------------------------------------------------------------------
# generate_regime_plots — agent-facing chart generation (audit Gate 3)
# ---------------------------------------------------------------------------
def test_generate_regime_plots_from_stored_fit(tmp_path):
    """fit (data_key + result_key) -> generate plots -> plots in the depot,
    exportable. The caller only ever handles short string keys."""
    import numpy as np

    import lazystats.regimes.db as rdb
    from lazystats.regimes import fit_regimes, generate_regime_plots
    from lazystats.regimes.tools import _swrite

    rdb.init_regime_db(str(tmp_path / "plots.db"))
    try:
        rng = np.random.RandomState(0)
        n = 200
        states = np.zeros(n, dtype=int)
        states[n // 3: 2 * n // 3] = 1
        y = np.where(states == 0, rng.normal(0, 0.5, n), rng.normal(0, 3.0, n))
        _swrite("plotdata", {
            "Y": y.reshape(-1, 1), "columns": ["SPY"],
            "index": [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]})

        fit_regimes(data_key="plotdata", series_names=["SPY"],
                    result_key="plotfit", S_max=2, n_starts=1, random_state=0)

        out = generate_regime_plots("plotfit")
        assert out["n_plots"] == 3          # 1 series + 2 barcodes
        assert out["series"] == ["SPY"]
        assert all(isinstance(k, str) and k for k in out["plot_keys"])

        listed = rdb.db_list_plots()
        assert listed["count"] >= 3

        target = tmp_path / "chart.png"
        exported = rdb.db_export_plot(out["plot_keys"][0], str(target))
        assert exported["success"] is True
        assert target.exists() and target.stat().st_size > 1000

        # missing data_key on an inline fit fails with guidance
        fit_regimes(data=[float(v) for v in y], series_names=["SPY"],
                    result_key="inlinefit", S_max=2, n_starts=1, random_state=0)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="data_key"):
            generate_regime_plots("inlinefit")
    finally:
        rdb._DB = None
