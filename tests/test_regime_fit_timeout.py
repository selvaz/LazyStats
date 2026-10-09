"""Spawn-safe doubles: no network, HUB or production depot is used here."""
from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from lazystats.regimes.config import RegimeConfig
from tests.fit_doubles import fake_fit, large_fit


def runner():
    spec = importlib.util.spec_from_file_location(
        "_timeout_runner", Path(__file__).resolve().parents[1] / "run_regime_daily.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hanging_child_is_reaped_within_limit(tmp_path):
    from lazystats.regimes import fit_process

    marker = tmp_path / "pid"
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(fit_process.FitTimeout):
        fit_process.run_fit_in_process(
            "HANG", timeout_seconds=10, fit_function=fake_fit, marker=str(marker))
    assert time.monotonic() - started < 12  # includes at most one second of reap grace
    assert marker.exists(), "the double must actually start before timing out"
    assert int(marker.read_text()) != os.getpid()
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_success_and_large_result_cross_process_boundary(tmp_path):
    from lazystats.regimes import fit_process

    marker = tmp_path / "pid"
    got = fit_process.run_fit_in_process(
        "BIL", timeout_seconds=30, fit_function=fake_fit, marker=str(marker))
    assert got["symbol"] == "BIL"
    assert int(marker.read_text()) != os.getpid()
    got = fit_process.run_fit_in_process(
        "BIL", timeout_seconds=30, fit_function=large_fit)
    assert len(got["chart"]) == 2_000_000


def test_real_child_fit_reads_only_the_explicit_synthetic_hub(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("market_data_hub")
    from lazystats.regimes.fit_process import run_fit_in_process

    db = tmp_path / "synthetic.duckdb"
    connection = duckdb.connect(str(db))
    try:
        connection.execute("CREATE TABLE v_returns(date DATE, symbol VARCHAR, log_return DOUBLE)")
        connection.execute("INSERT INTO v_returns SELECT DATE '2026-01-01' + CAST(i AS INTEGER), "
                           "'BIL', 0.0001 + 0.00001 * sin(i) FROM range(80) AS t(i)")
    finally:
        connection.close()
    before = db.read_bytes()
    result = run_fit_in_process("BIL", market_db=str(db), end="2026-10-01",
                                s_max=1, n_starts=1, timeout_seconds=30)
    assert result["diagnostics"]["n_states"] == 1
    assert len(result["readings"]) == 80
    assert all(not reading["is_high_vol"] for reading in result["readings"])
    assert all(reading["prob_high_vol"] == 0.0 for reading in result["readings"])
    assert db.read_bytes() == before, "the explicitly selected HUB must remain read-only"


@pytest.mark.parametrize("symbol, message", [
    ("ERROR", "ValueError: bad synthetic returns"), ("CRASH", "17")])
def test_child_faults_are_distinct_from_timeout(symbol, message):
    from lazystats.regimes import fit_process

    with pytest.raises(fit_process.FitProcessError, match=message):
        fit_process.run_fit_in_process(symbol, timeout_seconds=30, fit_function=fake_fit)


def test_job_continues_and_reports_timeout_in_both_reports(tmp_path):
    mod = runner()
    cfg = RegimeConfig(instruments=("HANG", "BIL"), windows=(), comparisons=(),
                       s_max=3, n_starts=20, random_state=123, retro_days=30)
    report = tmp_path / "charts.html"
    step = mod._make_fit_and_persist(
        cfg, depot_path=str(tmp_path / "depot.sqlite"), dry_run=False,
        report_path=report, generated="test", fit_timeout_seconds=10,
        fit_function=fake_fit)
    bundle = step(json.dumps({"symbols": list(cfg.instruments), "window": "full",
                              "variant": None, "start": "", "as_of": "2026-10-01",
                              "market_db": "synthetic", "production_db": "synthetic"}))
    assert [o["status"] for o in bundle["outcomes"]] == ["timeout", "ok"]
    failed = bundle["daily_payload"]["errors"][0]
    assert failed["symbol"] == "HANG" and failed["status"] == "timeout"
    assert "timeout" in report.read_text(encoding="utf-8")
    from lazystats.io.depot import ResultDepot

    depot = ResultDepot(str(tmp_path / "depot.sqlite"))
    try:
        failures = depot.list(series_key="regime:HANG")
        assert depot.load(failures[0]["result_id"])["payload"]["status"] == "timeout"
    finally:
        depot.close()
    bundle = mod._make_persist_report(depot_path=str(tmp_path / "depot.sqlite"),
                                      dry_run=False)(json.dumps(bundle))
    daily = mod.write_daily_report(bundle, depot_path=str(tmp_path / "depot.sqlite"),
                                  out_dir=tmp_path)
    assert '"status": "timeout"' in daily.read_text(encoding="utf-8")
    summary = mod.summarise(json.dumps(bundle))["summary"]
    assert (summary["fitted"], summary["failed"]) == (1, 1)
    assert summary["failures"][0]["status"] == "timeout"


@pytest.mark.parametrize("outcomes, expected", [
    ([{"status": "ok"}], 0),
    ([{"status": "timeout"}, {"status": "ok"}], 2),
    ([{"status": "error"}, {"status": "ok"}], 2),
    ([{"status": "error"}], 1),
    ([{"status": "timeout"}], 2),
    ([], 3),
])
def test_g6_exit_codes(outcomes, expected):
    assert runner().run_exit_code(outcomes) == expected


def test_cli_returns_degraded_for_timeout(monkeypatch):
    mod = runner()
    cfg = RegimeConfig(instruments=("HANG",), windows=(), comparisons=(),
                       s_max=3, n_starts=20, random_state=123, retro_days=30)
    from lazystats.regimes.config import Window

    monkeypatch.setattr(mod, "load_config", lambda _: replace(cfg, windows=(Window("full", None),)))
    monkeypatch.setattr(mod, "build_plan", lambda *a, **kw: None)
    bundle = {"outcomes": [{"status": "timeout"}], "summary": {"failed": 1}}

    class Envelope:
        error = None

        def text(self):
            return json.dumps(bundle)

    monkeypatch.setattr(mod, "Agent", lambda **kw: lambda arg: Envelope())
    monkeypatch.setattr("sys.argv", ["runner", "--config", "synthetic", "--window", "full",
                                     "--depot", "unused", "--market-db", "synthetic",
                                     "--production-db", "synthetic"])
    assert mod.main() == 2
