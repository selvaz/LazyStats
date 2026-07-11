"""io tests: depot provenance, local loaders, datahub loader (with a stubbed
hub — the real market_data_hub is exercised only when installed)."""

from __future__ import annotations

import pytest

from lazystats import ReturnDataset, return_volatility
from lazystats.io.depot import ResultDepot
from lazystats.io.local import returns_from_csv


def test_depot_round_trip_with_provenance(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "depot.sqlite"))
    ds = ReturnDataset(
        instruments=["ticker:SPY"],
        rows=[{"date": "2024-01-01", "ticker:SPY": 0.01},
              {"date": "2024-01-02", "ticker:SPY": 0.02}],
    )
    payload = return_volatility(ds, frequency="W")
    rid = depot.save(
        kind="report",
        produced_by="lazystats.core.return_volatility",
        instruments=ds.instruments,
        payload=payload,
        provenance={"source": "test", "frequency": "W",
                    "lazystats_version": "0.1.0",
                    "window": {"start": "2024-01-01", "end": "2024-01-02"}},
    )
    loaded = depot.load(rid)
    assert loaded is not None
    assert loaded["payload"]["volatility"]["ticker:SPY"]["observations"] == 2
    assert loaded["provenance"]["window"]["start"] == "2024-01-01"
    index = depot.list(produced_by="lazystats.core.return_volatility")
    assert index[0]["result_id"] == rid and "payload" not in index[0]
    assert depot.load("res_missing") is None
    depot.close()


def test_depot_refuses_missing_provenance(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "d.sqlite"))
    with pytest.raises(ValueError, match="provenance"):
        depot.save(kind="report", produced_by="x", instruments=[],
                   payload={}, provenance={})


def test_local_csv_loader(tmp_path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "date,ticker:SPY,ticker:TLT\n"
        "2024-01-01,0.01,-0.02\n"
        "2024-01-02,,0.01\n",
        encoding="utf-8",
    )
    ds = returns_from_csv(csv_path)
    assert ds.instruments == ["ticker:SPY", "ticker:TLT"]
    assert ds.rows[1]["ticker:SPY"] is None
    assert ds.metadata["notebook_only"] is True
    vol = return_volatility(ds)["volatility"]["ticker:TLT"]
    assert vol["observations"] == 2


def test_datahub_loader_validation() -> None:
    from lazystats.io import datahub

    with pytest.raises(ValueError, match="frequency"):
        datahub.load_returns("SPY", frequency="X")


def test_datahub_loader_against_real_hub(monkeypatch) -> None:
    """End-to-end through the real hub package with a stubbed extract layer."""
    mdh = pytest.importorskip("market_data_hub")
    import pandas as pd

    from lazystats.io import datahub

    def fake_extract_returns(symbols, start=None, end=None, frequency="D"):
        idx = pd.to_datetime(["2024-01-05", "2024-01-12"])
        frame = pd.DataFrame({s: [0.01, -0.02] for s in symbols}, index=idx)
        return frame, {"n_rows": 2, "used_returns_view": True}

    monkeypatch.setattr(mdh.extract, "extract_returns", fake_extract_returns)
    ds = datahub.load_returns("SPY,TLT", start="2024-01-01", frequency="W")
    assert ds.instruments == ["ticker:SPY", "ticker:TLT"]
    assert ds.rows[0] == {"date": "2024-01-05", "ticker:SPY": 0.01,
                          "ticker:TLT": 0.01}
    assert ds.metadata["return_kind"] == "log"
    assert ds.metadata["source"] == "market-data-hub"

    out = return_volatility(ds, frequency="W")
    assert out["volatility"]["ticker:SPY"]["observations"] == 2

    with pytest.raises(ValueError, match="only ticker"):
        datahub.load_returns("macro:FEDFUNDS")
    with pytest.raises(ValueError, match="duplicate"):
        datahub.load_returns("SPY,ticker:SPY")
