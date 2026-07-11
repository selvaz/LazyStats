# -*- coding: utf-8 -*-
"""Tests for parameter persistence + fixed-parameter inference (tool layer).

Covers:
  - fit_regimes auto-persisting a provenance-rich parameter record
  - regime_params_load (by params_key and by bare result_key)
  - apply_regime_params on new data (panel + joint), no refitting
  - JSON-serializability of the stored record
  - error handling (series mismatch, feature-count mismatch)
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import lazystats.regimes as lz
from lazystats.regimes.tools import _swrite, _sread, _params_store_key


def _two_regime(T, mu0, mu1, seed):
    rng = np.random.RandomState(seed)
    s = (rng.rand(T) < 0.5).astype(int)
    return np.where(s == 0, rng.normal(mu0, 0.5, T), rng.normal(mu1, 2.0, T))


@pytest.fixture
def panel_data():
    T = 700
    Y = np.column_stack([_two_regime(T, 0.0, 0.3, 1), _two_regime(T, 0.1, -0.2, 2)])
    cols = ["AAA", "BBB"]
    idx = [f"2020-{(i % 12) + 1:02d}-01" for i in range(T)]
    _swrite("pdata", {"Y": Y, "columns": cols, "index": idx})
    return "pdata", cols, T


class TestAutoPersistAndProvenance:
    def test_fit_regimes_returns_params_key(self, panel_data):
        dk, cols, T = panel_data
        out = lz.fit_regimes(data_key=dk, result_key="r1", model="panel",
                             S_max=3, n_starts=6, random_state=1)
        assert out["params_key"] == _params_store_key("r1")

    def test_record_layout_and_series(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="r2", model="panel",
                       S_max=3, n_starts=6, random_state=1)
        rec = lz.regime_params_load("r2::params")
        assert rec["layout"] == "panel"
        assert rec["series"] == cols
        assert set(rec["params_by_series"]) == set(cols)

    def test_provenance_captured(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="r3", model="panel",
                       S_max=3, n_starts=6, random_state=7)
        prov = lz.regime_params_load("r3::params")["provenance"]
        assert prov["data_key"] == dk
        assert prov["n_timesteps"] == T
        assert prov["n_series"] == len(cols)
        assert prov["date_start"] and prov["date_end"]
        assert prov["criterion"] == "bic"
        assert prov["random_state"] == 7

    def test_record_is_json_serializable(self, panel_data):
        dk, _, _ = panel_data
        lz.fit_regimes(data_key=dk, result_key="r4", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        rec = lz.regime_params_load("r4::params")
        rec2 = json.loads(json.dumps(rec))
        assert rec2["layout"] == "panel"

    def test_load_by_bare_result_key(self, panel_data):
        dk, _, _ = panel_data
        lz.fit_regimes(data_key=dk, result_key="r5", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        rec = lz.regime_params_load("r5")
        assert rec["layout"] == "panel"


class TestApplyParams:
    def test_apply_panel_on_new_data(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="ap1", model="panel",
                       S_max=3, n_starts=6, random_state=1)
        Ynew = np.column_stack([_two_regime(T, 0.0, 0.3, 99),
                                _two_regime(T, 0.1, -0.2, 98)])
        _swrite("nd1", {"Y": Ynew, "columns": cols, "index": [str(i) for i in range(T)]})
        ap = lz.apply_regime_params("ap1::params", data_key="nd1")
        assert set(ap["series"]) == set(cols)
        assert ap["applied_from"] == "ap1::params"
        for col in cols:
            assert "current_state" in ap["series"][col]

    def test_apply_matches_fit_on_same_data(self, panel_data):
        # Inference with the saved params on the SAME data must reproduce the
        # decoded current state from the original fit.
        dk, cols, T = panel_data
        out = lz.fit_regimes(data_key=dk, result_key="ap2", model="panel",
                             S_max=3, n_starts=6, random_state=1)
        ap = lz.apply_regime_params("ap2::params", data_key=dk)
        for col in cols:
            assert ap["series"][col]["current_state"] == out["series"][col]["current_state"]

    def test_apply_joint_reordered_columns_aligns_by_name(self, panel_data):
        # Joint model trained on ['AAA','BBB']; applying to ['BBB','AAA']
        # (same data, swapped column order) must give the SAME decoded path as
        # the in-order application, i.e. emissions stay tied to the right series.
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="jo1", model="joint_diag",
                       S_max=2, n_starts=6, random_state=1)
        Y = _sread(dk)["Y"]
        # in-order
        _swrite("nd_io", {"Y": Y, "columns": cols,
                          "index": [str(i) for i in range(T)]})
        ap_io = lz.apply_regime_params("jo1::params", data_key="nd_io")
        # swapped column order + matching swapped names
        Y_sw = Y[:, ::-1]
        cols_sw = cols[::-1]
        _swrite("nd_sw", {"Y": Y_sw, "columns": cols_sw,
                          "index": [str(i) for i in range(T)]})
        ap_sw = lz.apply_regime_params("jo1::params", data_key="nd_sw")
        # Per-series current state must match regardless of input column order.
        for col in cols:
            assert ap_io["series"][col]["current_state"] == ap_sw["series"][col]["current_state"]

    def test_apply_joint_wrong_series_names_raises(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="jo2", model="joint_diag",
                       S_max=2, n_starts=6, random_state=1)
        Y = _sread(dk)["Y"]
        # same width, but names do not match the trained series
        _swrite("nd_bad", {"Y": Y, "columns": ["XXX", "YYY"],
                           "index": [str(i) for i in range(T)]})
        with pytest.raises(ValueError):
            lz.apply_regime_params("jo2::params", data_key="nd_bad")

    def test_apply_joint_feature_mismatch_raises(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="ap3", model="joint_diag",
                       S_max=3, n_starts=6, random_state=1)
        bad = np.zeros((50, 1))  # joint expects 2 features
        _swrite("nd3", {"Y": bad, "columns": ["only"], "index": [str(i) for i in range(50)]})
        with pytest.raises(ValueError):
            lz.apply_regime_params("ap3::params", data_key="nd3")

    def test_apply_panel_missing_series_raises(self, panel_data):
        dk, cols, T = panel_data
        lz.fit_regimes(data_key=dk, result_key="ap4", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        _swrite("nd4", {"Y": np.zeros((30, 1)), "columns": ["ZZZ"],
                        "index": [str(i) for i in range(30)]})
        with pytest.raises(ValueError):
            lz.apply_regime_params("ap4::params", data_key="nd4")


class TestResave:
    def test_resave_under_new_key(self, panel_data):
        dk, _, _ = panel_data
        lz.fit_regimes(data_key=dk, result_key="rs1", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        r = lz.regime_params_save("rs1", params_key="rs1_backup")
        assert r["params_key"] == "rs1_backup"
        rec = lz.regime_params_load("rs1_backup")
        assert rec["layout"] == "panel"


class TestDiscovery:
    def test_list_finds_saved_model(self, panel_data):
        dk, cols, _ = panel_data
        lz.fit_regimes(data_key=dk, result_key="disc1", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        out = lz.regime_params_list()
        keys = {m["params_key"] for m in out["models"]}
        assert "disc1::params" in keys
        rec = next(m for m in out["models"] if m["params_key"] == "disc1::params")
        assert rec["data_key"] == dk
        assert rec["series"] == cols

    def test_list_filter_by_data_key(self, panel_data):
        dk, _, _ = panel_data
        lz.fit_regimes(data_key=dk, result_key="disc2", model="panel",
                       S_max=2, n_starts=6, random_state=1)
        hits = lz.regime_params_list(data_key=dk)["models"]
        assert all(m["data_key"] == dk for m in hits)
        misses = lz.regime_params_list(data_key="no_such_key")
        assert misses["count"] == 0


class TestSqliteDepot:
    """Exercise the indexed model_params table + schema migration."""

    def test_indexed_persistence_and_discovery(self, tmp_path):
        import lazystats.regimes.db as dbm
        from lazystats.regimes.tools import load_time_series
        import pandas as pd

        dbm.init_regime_db(str(tmp_path / "proj.db"))
        try:
            T = 400
            df = pd.DataFrame({"SPY": _two_regime(T, 0.0, 0.3, 1)},
                              index=pd.date_range("2020-01-01", periods=T, freq="D"))
            csv = tmp_path / "a.csv"
            df.to_csv(csv, index_label="date")
            load_time_series(file_path=str(csv), value_columns=["SPY"],
                             date_column="date", data_key="spy")
            lz.fit_regimes(data_key="spy", result_key="spy_reg", model="panel",
                           S_max=2, n_starts=6)
            # discovery via indexed table
            out = lz.regime_params_list(data_key="spy")
            assert out["count"] == 1
            assert out["models"][0]["params_key"] == "spy_reg::params"
            assert out["models"][0]["date_start"]  # provenance preserved
            # reload verbatim from the DB record
            rec = lz.regime_params_load("spy_reg::params")
            assert rec["provenance"]["data_key"] == "spy"
        finally:
            dbm._DB = None
            dbm._CACHE = {}

    def test_schema_v1_to_v2_migration(self, tmp_path):
        import lazystats.regimes.db as dbm
        p = str(tmp_path / "old.db")
        db = dbm.RegimeDB(p)
        with db._conn() as c:
            c.execute("DROP TABLE IF EXISTS model_params")
            c.execute("DELETE FROM schema_version")
            c.execute("INSERT INTO schema_version VALUES (1)")
        # reopening must add the table and bump the version
        db2 = dbm.RegimeDB(p)
        with db2._conn() as c:
            ver = c.execute("SELECT version FROM schema_version").fetchone()[0]
            has = c.execute(
                "SELECT name FROM sqlite_master WHERE name='model_params'").fetchone()
        assert ver == 2 and has is not None
