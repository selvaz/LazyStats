# -*- coding: utf-8 -*-
"""Hermetic tests for lazystats.regimes.datasources.load_from_datahub.

No market-data-hub install and no network access required: the loader's
lazily-imported ``extract_returns`` symbol is monkeypatched (patched where it
is looked up, i.e. inside ``lazystats.regimes.datasources.datahub``) to return a synthetic
(DataFrame, meta) pair built from the conftest factories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lazystats.regimes.datasources.datahub as datahub_mod
from lazystats.regimes import load_from_datahub
from lazystats.regimes.tools import _sread, _resolve_data, fit_regimes

from tests.regimes.conftest import make_multivariate_df


def _fake_extract_returns_factory(df: pd.DataFrame):
    """Return a stand-in for market_data_hub.extract_returns yielding (df, meta).

    Records the kwargs it was called with so the test can assert pass-through.
    """
    captured = {}

    def _fake(symbols, start=None, end=None, *, frequency="W",
              field="adj_close", fillna="none", db_path=None):
        captured["symbols"] = symbols
        captured["start"] = start
        captured["end"] = end
        captured["frequency"] = frequency
        captured["field"] = field
        captured["fillna"] = fillna
        captured["db_path"] = db_path
        meta = {
            "source": "market-data-hub",
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
        }
        return df, meta

    return _fake, captured


@pytest.fixture()
def synthetic_returns_df():
    """A small 2-column returns DataFrame with a DatetimeIndex."""
    df = make_multivariate_df(T=60, k=2, seed=7)
    # Mirror extract_returns output: a wide returns matrix, DatetimeIndex.
    df.columns = ["SPY", "TLT"]
    return df


def test_load_from_datahub_return_dict(monkeypatch, synthetic_returns_df):
    df = synthetic_returns_df
    fake, captured = _fake_extract_returns_factory(df)
    # Patch where the name is looked up (inside the loader module).
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    out = load_from_datahub(
        ["SPY", "TLT"], start="2010-01-01", end="2011-01-01",
        frequency="W", field="adj_close", data_key="dh_test",
    )

    assert out["data_key"] == "dh_test"
    assert out["n_rows"] == len(df)
    assert out["n_cols"] == 2
    assert out["columns"] == ["SPY", "TLT"]
    assert out["source"] == "market-data-hub"
    assert out["frequency"] == "W"
    assert out["field"] == "adj_close"
    assert len(out["date_range"]) == 2

    # kwargs were forwarded to extract_returns
    assert captured["symbols"] == ["SPY", "TLT"]
    assert captured["start"] == "2010-01-01"
    assert captured["frequency"] == "W"


def test_comma_separated_symbols_string_is_split(monkeypatch, synthetic_returns_df):
    # Over a tool/MCP boundary `symbols` can only arrive as a scalar string
    # (a JSON list is wrapped as a single element). A comma- or semicolon-
    # separated string must be split into a list before it reaches
    # extract_returns, otherwise it is treated as one (nonexistent) symbol.
    df = synthetic_returns_df
    fake, captured = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    out = load_from_datahub("SPY, TLT", data_key="dh_csv")  # note the stray space

    assert captured["symbols"] == ["SPY", "TLT"]  # split + stripped
    assert out["columns"] == ["SPY", "TLT"]
    assert out["n_cols"] == 2


def test_single_symbol_string_is_left_untouched(monkeypatch):
    # A bare token (no separator) must stay a plain string to preserve the
    # exact single-symbol behaviour extract_returns already handles.
    df = make_multivariate_df(T=40, k=1, seed=11)
    df.columns = ["SPY"]
    fake, captured = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    load_from_datahub("SPY", data_key="dh_single")

    assert captured["symbols"] == "SPY"


def test_payload_matches_load_time_series_shape(monkeypatch, synthetic_returns_df):
    df = synthetic_returns_df
    fake, _ = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    load_from_datahub(["SPY", "TLT"], data_key="dh_shape")

    payload = _sread("dh_shape")
    # Exact payload shape that load_time_series stores.
    assert set(payload.keys()) == {"Y", "columns", "index"}
    assert isinstance(payload["Y"], np.ndarray)
    assert payload["Y"].dtype == np.float64
    assert payload["Y"].shape == (len(df), 2)
    assert payload["columns"] == ["SPY", "TLT"]
    assert payload["index"] == [str(i) for i in df.index]

    # The standard consumer path resolves it cleanly.
    Y, cols = _resolve_data([], "dh_shape", [])
    assert Y.shape == (len(df), 2)
    assert cols == ["SPY", "TLT"]


def test_fit_regimes_consumes_data_key(monkeypatch):
    # Tiny series so the fit stays fast.
    df = make_multivariate_df(T=80, k=1, seed=3)
    df.columns = ["SPY"]
    fake, _ = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    load_from_datahub(["SPY"], data_key="dh_fit")

    result = fit_regimes(
        data_key="dh_fit", result_key="dh_fit_res",
        model="panel", S_max=2, n_starts=3, random_state=0,
    )

    assert result["result_key"] == "dh_fit_res"
    assert result["n_timesteps"] == len(df)
    assert "SPY" in result["series"]
    sd = result["series"]["SPY"]
    assert sd["S"] >= 1
    assert "current_label" in sd


def test_partial_missing_rows_are_dropped(monkeypatch, synthetic_returns_df):
    # Staggered history: TLT is missing for the first few rows. With fillna="none"
    # those rows have a NaN in one column and must be dropped before storing so
    # the stored matrix is finite and fit_regimes-consumable.
    df = synthetic_returns_df.copy()
    df.iloc[:5, df.columns.get_loc("TLT")] = np.nan
    fake, _ = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    out = load_from_datahub(["SPY", "TLT"], data_key="dh_partial")

    assert out["n_rows"] == len(df) - 5  # the 5 partially-missing rows are gone
    payload = _sread("dh_partial")
    assert not np.isnan(payload["Y"]).any()  # finite matrix, safe to fit
    assert payload["Y"].shape == (len(df) - 5, 2)


def test_all_missing_raises(monkeypatch, synthetic_returns_df):
    # If nothing survives the missing-value drop, fail loudly instead of storing
    # an empty/unusable matrix.
    df = synthetic_returns_df.copy()
    df.iloc[:, df.columns.get_loc("TLT")] = np.nan
    fake, _ = _fake_extract_returns_factory(df)
    monkeypatch.setattr(datahub_mod, "extract_returns", fake, raising=False)

    with pytest.raises(ValueError, match="complete data"):
        load_from_datahub(["SPY", "TLT"], data_key="dh_empty")


def test_missing_market_data_hub_raises(monkeypatch, synthetic_returns_df):
    # Force both import paths to fail so the helpful ImportError is raised.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "market_data_hub" or name.startswith("market_data_hub."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    # Ensure no module-level override shadows the import attempt.
    monkeypatch.setattr(datahub_mod, "extract_returns", None, raising=False)

    with pytest.raises(ImportError, match="market-data-hub"):
        load_from_datahub(["SPY"], data_key="dh_missing")
