"""Spawn-safe doubles: no network, HUB or production depot is used here."""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from lazystats.regimes.config import RegimeConfig
from tests.fit_doubles import changing_fit, fake_fit, large_fit
from tests.test_regime_worker_review import _alive


def _payload(symbols, *, as_of="2026-10-01", variant=None):
    return {"symbols": list(symbols), "window": "8y" if variant else "full",
            "variant": variant, "start": "", "as_of": as_of,
            "market_db": "synthetic", "production_db": "synthetic"}


def _run(mod, cfg, path, payload, **kwargs):
    return mod._make_fit_and_persist(
        cfg, depot_path=str(path), dry_run=False, report_path=kwargs.pop("report_path", None),
        generated="test", fit_timeout_seconds=kwargs.pop("fit_timeout_seconds", 30),
        fit_function=fake_fit, **kwargs)(json.dumps(payload))


def runner():
    spec = importlib.util.spec_from_file_location(
        "_timeout_runner", Path(__file__).resolve().parents[1] / "run_regime_daily.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hanging_child_is_reaped_within_limit(tmp_path):
    from lazystats.regimes import fit_process

    marker = tmp_path / "pid"
    started = time.monotonic()
    with pytest.raises(fit_process.FitTimeout):
        fit_process.run_fit_in_process(
            "HANG", timeout_seconds=10, fit_function=fake_fit, marker=str(marker))
    assert time.monotonic() - started < 12  # includes at most one second of reap grace
    assert marker.exists(), "the double must actually start before timing out"
    assert int(marker.read_text()) != os.getpid()
    assert not _alive(int(marker.read_text()))


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


def test_hanging_fit_is_bounded_worker_restarts_and_next_symbol_fits(tmp_path):
    mod = runner()
    cfg = RegimeConfig(instruments=("HANG", "BIL"), windows=(), comparisons=(),
                       s_max=1, n_starts=1, random_state=123, retro_days=30)
    started = time.monotonic()
    result = _run(mod, cfg, tmp_path / "depot.sqlite", _payload(cfg.instruments),
                  fit_timeout_seconds=1)
    assert time.monotonic() - started < 3
    assert [item["status"] for item in result["outcomes"]] == ["timeout", "ok"]


@pytest.mark.parametrize("old_status", ["timeout", "error"])
def test_resume_refits_today_failed_rows(tmp_path, old_status):
    from lazystats.io.depot import ResultDepot
    from lazystats.regimes.persist import write_failure

    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(), s_max=1,
                       n_starts=1, random_state=123, retro_days=30)
    path = tmp_path / "depot.sqlite"
    depot = ResultDepot(str(path))
    write_failure(depot, symbol="BIL", series_key="regime:BIL",
                  estimation_date="2026-10-01", error="old failure", status=old_status)
    depot.close()
    result = _run(mod, cfg, path, _payload(cfg.instruments))
    assert result["outcomes"][0]["status"] == "ok"


def test_ok_for_other_date_does_not_skip(tmp_path):
    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(), s_max=1,
                       n_starts=1, random_state=123, retro_days=30)
    path = tmp_path / "depot.sqlite"
    _run(mod, cfg, path, _payload(cfg.instruments, as_of="2026-09-30"))
    result = _run(mod, cfg, path, _payload(cfg.instruments))
    assert result["outcomes"][0]["status"] == "ok"


@pytest.mark.parametrize("first_variant,second_variant", [("8y", None), (None, "8y")])
def test_variants_do_not_skip_each_other(tmp_path, first_variant, second_variant):
    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(), s_max=1,
                       n_starts=1, random_state=123, retro_days=30)
    path = tmp_path / "depot.sqlite"
    _run(mod, cfg, path, _payload(cfg.instruments, variant=first_variant))
    result = _run(mod, cfg, path, _payload(cfg.instruments, variant=second_variant))
    assert result["outcomes"][0]["status"] == "ok"


def test_phase_log_has_documented_format(tmp_path, capsys):
    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(), s_max=1,
                       n_starts=1, random_state=123, retro_days=30)
    _run(mod, cfg, tmp_path / "depot.sqlite", _payload(cfg.instruments))
    line = capsys.readouterr().err.splitlines()[-2]
    assert line.startswith("regime fit symbol=BIL status=ok seconds=")
    assert " phases=[" in line and "fit_and_chart=" in line and line.endswith("s]")


def test_final_summary_log_has_documented_format(capsys):
    runner()._log_summary({"fitted": 2, "skipped": 1, "timeouts": 1,
                           "failed": 1, "total_seconds": 3.25})
    assert capsys.readouterr().err == (
        "regime summary fitted=2 skipped=1 timeouts=1 failures=1 total_seconds=3.250\n")


def test_deadline_crossed_waiting_for_write_rolls_back_ok_and_records_timeout(tmp_path):
    from lazystats.io.depot import ResultDepot
    from lazystats.regimes.persist import write_failure, write_fit

    path = tmp_path / "depot.sqlite"
    ResultDepot(str(path)).close()
    entered = threading.Event()

    def lock_briefly():
        holder = ResultDepot(str(path))
        with holder.transaction():
            entered.set()
            time.sleep(0.15)
        holder.close()

    thread = threading.Thread(target=lock_briefly)
    thread.start()
    assert entered.wait(1)
    depot = ResultDepot(str(path), busy_timeout_seconds=1)
    with pytest.raises(TimeoutError, match="before depot commit"):
        write_fit(depot, symbol="BIL", series_key="regime:BIL",
                  estimation_date="2026-10-01", diagnostics={"n_states": 1},
                  dates=["2026-10-01"], readings=[{"state": 0, "n_states": 1}],
                  retro_days=30, deadline=time.monotonic() + 0.05)
    thread.join()
    write_failure(depot, symbol="BIL", series_key="regime:BIL",
                  estimation_date="2026-10-01", error="deadline passed", status="timeout")
    rows = [depot.load(row["result_id"])["payload"]
            for row in depot.list(series_key="regime:BIL")]
    depot.close()
    assert [row["status"] for row in rows] == ["timeout"]


def test_daily_run_resumes_successes_and_force_refits(tmp_path):
    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(),
                       s_max=1, n_starts=1, random_state=123, retro_days=30)
    depot_path = str(tmp_path / "depot.sqlite")
    payload = {"symbols": ["BIL"], "window": "full", "variant": None, "start": "",
               "as_of": "2026-10-01", "market_db": "synthetic",
               "production_db": "synthetic"}

    def run(*, force=False):
        return mod._make_fit_and_persist(
            cfg, depot_path=depot_path, dry_run=False, report_path=None, generated="test",
            fit_timeout_seconds=30, fit_function=fake_fit, force=force)(json.dumps(payload))

    first, resumed, forced = run(), run(), run(force=True)
    assert [first["outcomes"][0]["status"], resumed["outcomes"][0]["status"],
            forced["outcomes"][0]["status"]] == ["ok", "skipped", "ok"]
    summary = mod.summarise(json.dumps(resumed))["summary"]
    assert (summary["fitted"], summary["skipped"], summary["timeouts"],
            summary["failed"]) == (0, 1, 0, 0)


def test_resumed_report_rebuilds_skipped_half_with_change_flags(tmp_path):
    mod = runner()
    path = str(tmp_path / "depot.sqlite")
    common = dict(windows=(), comparisons=(), s_max=1, n_starts=1,
                  random_state=123, retro_days=30)
    first_cfg = RegimeConfig(instruments=("CHANGED",), **common)
    full_cfg = replace(first_cfg, instruments=("CHANGED", "STEADY"))

    def run(cfg):
        payload = {"symbols": list(cfg.instruments), "window": "full", "variant": None,
                   "start": "", "as_of": "2026-10-01", "market_db": "synthetic",
                   "production_db": "synthetic"}
        return mod._make_fit_and_persist(
            cfg, depot_path=path, dry_run=False, report_path=None, generated="test",
            fit_timeout_seconds=30, fit_function=changing_fit)(json.dumps(payload))

    run(first_cfg)
    resumed = run(full_cfg)
    assert [item["status"] for item in resumed["outcomes"]] == ["skipped", "ok"]
    records = {item["symbol"]: item for item in resumed["daily_payload"]["symbols"]}
    assert set(records) == {"CHANGED", "STEADY"}
    assert records["CHANGED"]["changed_today"] is True
    assert records["STEADY"]["changed_today"] is False
    assert resumed["daily_payload"]["summary"]["n_changed_today"] == 1


def test_skipped_symbol_chart_report_explicitly_identifies_stored_result(tmp_path):
    mod = runner()
    cfg = RegimeConfig(instruments=("BIL",), windows=(), comparisons=(), s_max=1,
                       n_starts=1, random_state=123, retro_days=30)
    path = tmp_path / "depot.sqlite"
    _run(mod, cfg, path, _payload(cfg.instruments))
    report = tmp_path / "report.html"
    result = _run(mod, cfg, path, _payload(cfg.instruments), report_path=report)
    assert result["outcomes"][0]["status"] == "skipped"
    assert "Not refitted today (stored result from" in report.read_text(encoding="utf-8")


def test_snapshot_read_path_reaches_worker_without_changing_series_identity(
        tmp_path, monkeypatch):
    mod = runner()
    cfg = RegimeConfig(instruments=("GLD",), windows=(), comparisons=(),
                       s_max=1, n_starts=1, random_state=123, retro_days=30)
    calls = []

    class Worker:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def fit(self, symbol, **kwargs):
            calls.append((symbol, kwargs))
            return {
                "symbol": symbol,
                "dates": ["2026-10-01"],
                "readings": [{"state": 0, "is_high_vol": False,
                              "prob_high_vol": 0.0, "state_probs": [1.0]}],
                "diagnostics": {"n_states": 1, "states": [], "labels": [],
                                "transmat": [], "n_obs": 1},
                "chart": None,
            }

    monkeypatch.setattr(mod, "FitWorker", Worker)
    step = mod._make_fit_and_persist(
        cfg, depot_path=str(tmp_path / "depot.sqlite"), dry_run=True,
        report_path=None, generated="test")
    result = step(json.dumps({"symbols": ["GLD"], "window": "full", "variant": None,
                              "start": "", "as_of": "2026-10-01",
                              "market_db": "production.duckdb",
                              "market_read_db": "snapshot.duckdb",
                              "production_db": "production.duckdb"}))

    assert result["outcomes"][0]["series_key"] == "regime:GLD"
    assert calls[0][1]["market_db"] == "production.duckdb"
    assert calls[0][1]["market_read_db"] == "snapshot.duckdb"


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
