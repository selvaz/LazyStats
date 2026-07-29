"""
Tests for regime_db.py — SQLite persistence layer.

Each test gets its own isolated in-memory-backed database via a fixture.
No network calls, no shared global state.

Run with:  pytest tests/test_db.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import lazystats.regimes.db as rdb
from lazystats.regimes.db import RegimeDB, _numpy_to_blob, _blob_to_numpy, _df_to_blobs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh RegimeDB backed by a temporary SQLite file."""
    return RegimeDB(str(tmp_path / "test_regime.db"))


@pytest.fixture
def simple_df():
    """Small deterministic DataFrame for round-trip tests."""
    rng = np.random.RandomState(0)
    idx = pd.date_range("2020-01-01", periods=50, freq="W")
    return pd.DataFrame({"A": rng.randn(50), "B": rng.randn(50)}, index=idx)


# ---------------------------------------------------------------------------
# Group 1: Serialization helpers
# ---------------------------------------------------------------------------

class TestNumpyBlob:
    def test_roundtrip_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        blob = _numpy_to_blob(arr)
        arr_back = _blob_to_numpy(blob)
        np.testing.assert_array_equal(arr, arr_back)

    def test_roundtrip_2d(self):
        arr = np.random.RandomState(0).randn(10, 3)
        blob = _numpy_to_blob(arr)
        arr_back = _blob_to_numpy(blob)
        np.testing.assert_allclose(arr, arr_back, rtol=1e-12)

    def test_roundtrip_int_array(self):
        arr = np.arange(20, dtype=np.int32)
        blob = _numpy_to_blob(arr)
        arr_back = _blob_to_numpy(blob)
        np.testing.assert_array_equal(arr.astype(float), arr_back)

    def test_blob_is_bytes(self):
        arr = np.ones((5, 2))
        assert isinstance(_numpy_to_blob(arr), bytes)


class TestDfToBlobs:
    def test_roundtrip_preserves_values(self, simple_df):
        blob, col_json, idx_json = _df_to_blobs(simple_df)
        arr = _blob_to_numpy(blob)
        cols = json.loads(col_json)
        assert arr.shape == (50, 2)
        assert cols == ["A", "B"]

    def test_columns_json_is_list(self, simple_df):
        _, col_json, _ = _df_to_blobs(simple_df)
        cols = json.loads(col_json)
        assert isinstance(cols, list)

    def test_index_json_is_list(self, simple_df):
        _, _, idx_json = _df_to_blobs(simple_df)
        idx = json.loads(idx_json)
        assert isinstance(idx, list)
        assert len(idx) == 50


# ---------------------------------------------------------------------------
# Group 2: Time series I/O
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def test_write_and_exists(self, db, simple_df):
        db.write_series("ts_test", simple_df)
        assert db.series_exists("ts_test")

    def test_nonexistent_key_returns_false(self, db):
        assert not db.series_exists("no_such_key")

    def test_read_series_returns_dict(self, db, simple_df):
        db.write_series("ts_read", simple_df)
        result = db.read_series("ts_read")
        assert result is not None
        assert "Y" in result
        assert "columns" in result
        assert "index" in result

    def test_read_preserves_shape(self, db, simple_df):
        db.write_series("ts_shape", simple_df)
        result = db.read_series("ts_shape")
        assert result["Y"].shape == (50, 2)

    def test_read_preserves_columns(self, db, simple_df):
        db.write_series("ts_cols", simple_df)
        result = db.read_series("ts_cols")
        assert result["columns"] == ["A", "B"]

    def test_read_nonexistent_returns_none(self, db):
        assert db.read_series("absent_key") is None

    def test_list_series_returns_all_keys(self, db, simple_df):
        db.write_series("key_1", simple_df)
        db.write_series("key_2", simple_df)
        listed = [r["data_key"] for r in db.list_series()]
        assert "key_1" in listed
        assert "key_2" in listed

    def test_list_series_empty_initially(self, db):
        assert db.list_series() == []

    def test_delete_series_removes_key(self, db, simple_df):
        db.write_series("del_me", simple_df)
        assert db.series_exists("del_me")
        removed = db.delete_series("del_me")
        assert removed is True
        assert not db.series_exists("del_me")

    def test_delete_nonexistent_returns_false(self, db):
        assert db.delete_series("ghost_key") is False

    def test_write_replace_existing(self, db, simple_df):
        db.write_series("overwrite_me", simple_df)
        new_df = simple_df * 2
        db.write_series("overwrite_me", new_df)  # should not raise
        result = db.read_series("overwrite_me")
        np.testing.assert_allclose(result["Y"], new_df.values, rtol=1e-10)

    def test_get_series_info_returns_metadata(self, db, simple_df):
        db.write_series("info_test", simple_df, source="computed")
        info = db.get_series_info("info_test")
        assert info is not None
        assert info["data_key"] == "info_test"
        assert info["n_rows"] == 50
        assert info["n_cols"] == 2
        assert info["source"] == "computed"

    def test_get_series_info_nonexistent_returns_none(self, db):
        assert db.get_series_info("missing") is None

    def test_single_column_dataframe(self, db):
        df = pd.DataFrame({"X": np.ones(20)},
                           index=pd.date_range("2021-01-01", periods=20))
        db.write_series("single_col", df)
        res = db.read_series("single_col")
        assert res["Y"].shape == (20, 1)
        assert res["columns"] == ["X"]


# ---------------------------------------------------------------------------
# Group 3: Generic key-value store
# ---------------------------------------------------------------------------

class TestGenericKV:
    def test_write_and_read_dict(self, db):
        data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
        db.generic_write("kv_dict", data)
        result = db.generic_read("kv_dict")
        assert result == data

    def test_write_and_read_list(self, db):
        lst = [1, 2.5, "three"]
        db.generic_write("kv_list", lst)
        assert db.generic_read("kv_list") == lst

    def test_write_and_read_scalar(self, db):
        db.generic_write("kv_float", 3.14)
        assert abs(db.generic_read("kv_float") - 3.14) < 1e-10

    def test_read_missing_key_raises_key_error(self, db):
        with pytest.raises(KeyError):
            db.generic_read("no_such_kv_key")

    def test_generic_list_returns_all_keys(self, db):
        db.generic_write("gk_1", 1)
        db.generic_write("gk_2", 2)
        keys = db.generic_list()
        assert "gk_1" in keys
        assert "gk_2" in keys

    def test_generic_list_empty_initially(self, db):
        assert db.generic_list() == []

    def test_overwrite_updates_value(self, db):
        db.generic_write("update_me", {"v": 1})
        db.generic_write("update_me", {"v": 99})
        result = db.generic_read("update_me")
        assert result["v"] == 99


# ---------------------------------------------------------------------------
# Group 4: Module-level API (init_regime_db, swrite, sread, slist)
# ---------------------------------------------------------------------------

class TestModuleLevelAPI:
    @pytest.fixture(autouse=True)
    def fresh_db(self, tmp_path):
        """Reinitialize module-level state for each test."""
        rdb.init_regime_db(str(tmp_path / "mod_test.db"))

    def test_get_db_after_init(self):
        db = rdb.get_db()
        assert isinstance(db, RegimeDB)

    def test_swrite_sread_roundtrip(self):
        rdb.swrite("mod_key", {"x": 42})
        result = rdb.sread("mod_key")
        assert result == {"x": 42}

    def test_sread_missing_raises(self):
        with pytest.raises(KeyError):
            rdb.sread("not_there")

    def test_slist_contains_written_key(self):
        rdb.swrite("listed_key", "hello")
        assert "listed_key" in rdb.slist()

    def test_slist_returns_sorted_list(self):
        rdb.swrite("z_key", 1)
        rdb.swrite("a_key", 2)
        keys = rdb.slist()
        assert keys == sorted(keys)

    def test_get_db_before_init_raises(self, tmp_path):
        # Reset global state
        rdb._DB = None
        with pytest.raises(RuntimeError, match="not initialized"):
            rdb.get_db()


class TestResolveDepotPath:
    """resolve_depot_path is the single resolution chain every caller (LazyTools'
    RegimeTools, the MCP server's regimes/report providers) must route through --
    two independent copies of this logic is exactly how a caller's explicit
    init_regime_db() got silently overridden by a different default elsewhere."""

    @pytest.fixture(autouse=True)
    def _reset_active_depot(self):
        # _DB is a process-wide global (see Group 4 below) -- reset it around
        # every test in this class so "no active depot" tests aren't polluted
        # by whatever ran earlier in the same pytest session, and so a test
        # that does init_regime_db() doesn't leak into the next one.
        rdb._DB = None
        yield
        rdb._DB = None

    def test_explicit_wins_over_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAZYTOOLS_REGIME_DB", str(tmp_path / "env.db"))
        rdb.init_regime_db(str(tmp_path / "active.db"))
        explicit = str(tmp_path / "explicit.db")
        assert rdb.resolve_depot_path(explicit) == explicit

    def test_active_depot_wins_over_env_and_default(self, monkeypatch, tmp_path):
        """The exact regression an earlier version of this resolver reintroduced
        one level down: init_regime_db("/custom/path.db") already ran, then a
        DIFFERENT caller asks for the default (no explicit path) -- it must get
        back the depot that's already running, never a freshly recomputed
        env/default that would silently redirect it to another file."""
        monkeypatch.setenv("LAZYTOOLS_REGIME_DB", str(tmp_path / "env.db"))
        active_path = str(tmp_path / "active.db")
        rdb.init_regime_db(active_path)
        assert rdb.resolve_depot_path() == active_path

    def test_env_var_wins_when_no_explicit_and_no_active_depot(self, monkeypatch, tmp_path):
        env_path = str(tmp_path / "env.db")
        monkeypatch.setenv("LAZYTOOLS_REGIME_DB", env_path)
        assert rdb.resolve_depot_path() == env_path

    def test_falls_back_to_data_dir_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LAZYTOOLS_REGIME_DB", raising=False)
        monkeypatch.setenv("LAZYTOOLS_DATA_DIR", str(tmp_path))
        resolved = rdb.resolve_depot_path()
        assert resolved == str(tmp_path / "regime_depot.db")
        assert tmp_path.is_dir()  # the data dir must actually exist afterwards

    def test_falls_back_to_home_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("LAZYTOOLS_REGIME_DB", raising=False)
        monkeypatch.delenv("LAZYTOOLS_DATA_DIR", raising=False)
        import os
        resolved = rdb.resolve_depot_path()
        assert resolved == os.path.join(os.path.expanduser("~"), ".lazytools", "regime_depot.db")
