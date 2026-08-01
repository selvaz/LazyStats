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


def test_depot_save_stable_requires_series_key(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "stable_requires_key.sqlite"))
    with pytest.raises(ValueError, match="series_key"):
        depot.save(kind="regime", produced_by="x", instruments=[],
                   payload={}, provenance={"source": "test"}, cadence="stable")
    depot.close()


def test_depot_save_adhoc_default_unchanged(tmp_path) -> None:
    """cadence defaults to 'adhoc' and existing callers keep working exactly
    as before: no series_key required, and load()/list() gain the new
    cadence/series_key fields without breaking the old ones."""
    depot = ResultDepot(str(tmp_path / "adhoc.sqlite"))
    rid = depot.save(
        kind="report", produced_by="lazystats.core.return_volatility",
        instruments=["ticker:SPY"], payload={"a": 1},
        provenance={"source": "test"},
    )
    loaded = depot.load(rid)
    assert loaded["cadence"] == "adhoc"
    assert loaded["series_key"] is None
    index = depot.list(produced_by="lazystats.core.return_volatility")
    assert index[0]["cadence"] == "adhoc"
    assert index[0]["series_key"] is None
    depot.close()


def test_depot_save_stable_with_series_key(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "stable.sqlite"))
    rid = depot.save(
        kind="regime", produced_by="lazyhmm.fit", instruments=["ticker:SPY"],
        payload={"n_states": 2}, provenance={"source": "test"},
        cadence="stable", series_key="spy_regime_daily",
    )
    loaded = depot.load(rid)
    assert loaded["cadence"] == "stable"
    assert loaded["series_key"] == "spy_regime_daily"
    index = depot.list(cadence="stable")
    assert index[0]["result_id"] == rid
    assert depot.list(cadence="adhoc") == []
    depot.close()


def test_depot_save_stable_point_append_on_change(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "points.sqlite"))
    series_key = "spy_regime_daily"

    inserted = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-01", value={"state": 0, "label": "low_vol"},
    )
    assert inserted is True

    # Same value, same estimation_date re-run -> no-op.
    inserted_again = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-01", value={"state": 0, "label": "low_vol"},
    )
    assert inserted_again is False

    # Same value, later estimation_date -> still unchanged, no-op.
    unchanged_later = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-02", value={"state": 0, "label": "low_vol"},
    )
    assert unchanged_later is False

    # Genuinely different value, new estimation_date -> revision inserted.
    revised = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-03", value={"state": 1, "label": "high_vol"},
    )
    assert revised is True

    depot.close()


def test_depot_save_stable_point_same_day_rerun_replaces_not_crashes(tmp_path) -> None:
    """A same-day rerun (identical estimation_date) whose refit produced a
    different value must replace that row in place, not raise a UNIQUE
    constraint violation -- this is the same estimation event, not a new
    vintage. Regression test: a plain INSERT here previously crashed with
    sqlite3.IntegrityError on the (series_key, as_of_date, estimation_date)
    PRIMARY KEY."""
    depot = ResultDepot(str(tmp_path / "rerun.sqlite"))
    series_key = "spy_regime_daily"

    first = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-01", value={"state": 0},
    )
    assert first is True

    rerun = depot.save_stable_point(
        series_key=series_key, as_of_date="2024-01-01",
        estimation_date="2024-01-01", value={"state": 1},
    )
    assert rerun is True

    vintages = depot.list_series_vintages(series_key, "2024-01-01")
    assert len(vintages) == 1, "must replace in place, not accumulate a duplicate vintage row"
    assert vintages[0]["value"]["state"] == 1

    depot.close()


def test_depot_list_series_vintages(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "vintages.sqlite"))
    series_key = "spy_regime_daily"

    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-01", value={"state": 0})
    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-02", value={"state": 1})
    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-03", value={"state": 1})  # no-op
    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-04", value={"state": 2})

    vintages = depot.list_series_vintages(series_key, "2024-01-01")
    assert [v["estimation_date"] for v in vintages] == ["2024-01-01", "2024-01-02", "2024-01-04"]
    assert [v["value"]["state"] for v in vintages] == [0, 1, 2]
    depot.close()


def test_depot_get_series_latest(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "latest.sqlite"))
    series_key = "spy_regime_daily"

    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-01", value={"state": 0})
    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-01",
                            estimation_date="2024-01-03", value={"state": 1})
    depot.save_stable_point(series_key=series_key, as_of_date="2024-01-02",
                            estimation_date="2024-01-02", value={"state": 1})

    latest = depot.get_series_latest(series_key)
    assert [row["as_of_date"] for row in latest] == ["2024-01-01", "2024-01-02"]
    assert latest[0]["estimation_date"] == "2024-01-03"
    assert latest[0]["value"]["state"] == 1  # the revised value, not the original 0
    assert latest[1]["value"]["state"] == 1

    filtered = depot.get_series_latest(series_key, since="2024-01-02")
    assert [row["as_of_date"] for row in filtered] == ["2024-01-02"]
    depot.close()


def test_depot_save_and_get_detail(tmp_path) -> None:
    depot = ResultDepot(str(tmp_path / "detail.sqlite"))
    rid = depot.save(kind="ols", produced_by="x", instruments=[],
                     payload={}, provenance={"source": "test"})

    assert depot.get_detail(rid, "residuals") is None

    depot.save_detail(rid, "residuals", b"first-blob")
    assert depot.get_detail(rid, "residuals") == b"first-blob"

    depot.save_detail(rid, "residuals", b"second-blob")
    assert depot.get_detail(rid, "residuals") == b"second-blob"

    depot.save_detail(rid, "predictions", b"other-detail-type")
    assert depot.get_detail(rid, "predictions") == b"other-detail-type"
    assert depot.get_detail(rid, "residuals") == b"second-blob"

    depot.close()


def test_depot_migration_idempotent(tmp_path) -> None:
    """Re-opening an already-migrated depot file must not error, and must not
    duplicate or lose the v2 columns."""
    path = str(tmp_path / "reopen.sqlite")
    depot1 = ResultDepot(path)
    rid = depot1.save(kind="report", produced_by="x", instruments=[],
                      payload={}, provenance={"source": "test"},
                      cadence="stable", series_key="k")
    depot1.close()

    depot2 = ResultDepot(path)
    loaded = depot2.load(rid)
    assert loaded["cadence"] == "stable"
    assert loaded["series_key"] == "k"
    cols = {row[1] for row in depot2._con.execute(
        "PRAGMA table_info(analysis_results)").fetchall()}
    assert cols == {"result_id", "kind", "produced_by", "instruments",
                    "payload", "provenance", "created_at", "cadence", "series_key"}
    depot2.close()


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
