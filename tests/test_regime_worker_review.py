"""Reusable worker regressions and an offline Windows overhead benchmark."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fit_doubles import fake_fit, large_fit, recording_fit

ROOT = Path(__file__).resolve().parents[1]


def test_child_error_keeps_full_traceback(tmp_path):
    from lazystats._fit_worker import fit_worker
    from lazystats.regimes.fit_process import FitProcessError, FitWorker

    path = tmp_path / "error.json"
    fit_worker(str(path), "ERROR", {}, fake_fit)
    error = json.loads(path.read_text())["error"]
    assert "Traceback (most recent call last)" in error
    assert "fit_doubles.py" in error
    assert "ValueError: bad synthetic returns" in error
    with FitWorker() as worker:
        with pytest.raises(FitProcessError) as caught:
            worker.fit("ERROR", fit_function=fake_fit, timeout_seconds=30)
        assert "Traceback (most recent call last)" in str(caught.value)
        assert "fit_doubles.py" in str(caught.value)
        before = worker.process
        assert worker.fit("BIL", fit_function=fake_fit)["symbol"] == "BIL"
        assert worker.process is before


def test_worker_reuses_pid_and_warm_overhead_is_under_one_second(tmp_path):
    from lazystats.regimes.fit_process import FitWorker

    durations, pids = [], []
    with FitWorker() as worker:
        for i in range(4):
            marker = tmp_path / str(i)
            started = time.perf_counter()
            assert worker.fit("BIL", fit_function=fake_fit, marker=str(marker),
                              timeout_seconds=30)["symbol"] == "BIL"
            durations.append(time.perf_counter() - started)
            pids.append(int(marker.read_text()))
        assert len(set(pids)) == 1
        if sys.platform == "win32":
            assert worker._job is not None, "the Windows Job Object must be active"
        process = worker.process
        assert all(t < 1 for t in durations[1:]), durations
        assert len(worker.fit("BIL", fit_function=large_fit,
                              timeout_seconds=30)["chart"]) == 2_000_000
    assert process.poll() is not None
    print("worker request seconds:", ", ".join(f"{t:.6f}" for t in durations))


@pytest.mark.parametrize("failure", ["HANG", "HOLD_GIL", "CRASH"])
def test_worker_restarts_after_timeout_or_crash(tmp_path, failure):
    from lazystats.regimes.fit_process import FitProcessError, FitTimeout, FitWorker

    marker = tmp_path / "pid"
    with FitWorker() as worker:
        worker.fit("BIL", fit_function=fake_fit, marker=str(marker), timeout_seconds=30)
        before = worker.process
        started = time.monotonic()
        with pytest.raises(FitProcessError if failure == "CRASH" else FitTimeout):
            worker.fit(failure, fit_function=fake_fit, timeout_seconds=1)
        assert time.monotonic() - started < 2
        assert before.poll() is not None
        worker.fit("BIL", fit_function=fake_fit, marker=str(marker), timeout_seconds=30)
        assert int(marker.read_text()) != before.pid
        after = worker.process
    assert after.poll() is not None


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT),
                                       env.get("PYTHONPATH", "")])
    return env


def _alive(pid):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        dll.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        dll.OpenProcess.restype = wintypes.HANDLE
        dll.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        dll.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = dll.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return dll.WaitForSingleObject(handle, 0) == 258
        finally:
            dll.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split()[2] == "Z")


@pytest.mark.parametrize("use_job", [True, False])
@pytest.mark.parametrize("symbol", ["HANG", "HOLD_GIL"])
def test_worker_dies_when_parent_is_killed_during_fit(tmp_path, use_job, symbol):
    from lazystats.regimes.fit_process import FitWorker  # noqa: F401

    marker = tmp_path / "pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "from lazystats.regimes.fit_process import FitWorker\n"
        "from tests.fit_doubles import fake_fit\n"
        f"with FitWorker(use_job_object={use_job!r}) as worker:\n"
        f"    worker.fit({symbol!r}, fit_function=fake_fit, marker={str(marker)!r}, "
        "timeout_seconds=60)\n", encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script)], env=_env(), cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline and parent.poll() is None:
            time.sleep(0.05)
        assert marker.exists(), "fit must be running before killing its parent"
        pid = int(marker.read_text())
        assert _alive(pid)
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), "a CPU-bound worker survived its killed parent"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if pid is not None and _alive(pid):
            import signal

            os.kill(pid, signal.SIGTERM)


def test_worker_does_not_reimport_cli_main(tmp_path):
    from lazystats.regimes.fit_process import FitWorker  # noqa: F401

    marker = tmp_path / "main-imports"
    script = tmp_path / "cli.py"
    script.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(marker)!r})\n"
        "with p.open('a') as stream: stream.write('imported\\n')\n"
        "from lazystats.regimes.fit_process import FitWorker\n"
        "from tests.fit_doubles import fake_fit\n"
        "if __name__ == '__main__':\n"
        "    with FitWorker() as worker:\n"
        "        for i in range(3): worker.fit('BIL', fit_function=fake_fit)\n",
        encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], env=_env(), cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert marker.read_text().splitlines() == ["imported"]


def test_daily_symbol_loop_reuses_one_worker_and_closes_it(tmp_path, monkeypatch):
    from lazystats.regimes.config import RegimeConfig

    spec = importlib.util.spec_from_file_location(
        "_worker_review_runner", ROOT / "run_regime_daily.py")

    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    marker = tmp_path / "pids"
    monkeypatch.setenv("LAZYSTATS_TEST_PID_LOG", str(marker))
    cfg = RegimeConfig(instruments=("BIL", "GLD", "SPY"), windows=(), comparisons=(),
                       s_max=1, n_starts=1, random_state=123, retro_days=30)
    step = runner._make_fit_and_persist(
        cfg, depot_path=str(tmp_path / "depot.sqlite"), dry_run=True,
        report_path=None, generated="test", fit_timeout_seconds=30,
        fit_function=recording_fit)
    result = step(json.dumps({"symbols": list(cfg.instruments), "window": "full",
                              "variant": None, "start": "", "as_of": "2026-10-01",
                              "market_db": "synthetic", "production_db": "synthetic"}))
    assert [o["status"] for o in result["outcomes"]] == ["ok"] * 3
    pids = marker.read_text().splitlines()
    assert len(pids) == 3 and len(set(pids)) == 1
    assert not _alive(int(pids[0]))
