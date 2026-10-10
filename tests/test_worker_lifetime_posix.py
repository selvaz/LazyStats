import os

from lazystats import _worker_lifetime


def _proc(tmp_path, pid, state):
    entry = tmp_path / str(pid)
    entry.mkdir()
    (entry / "stat").write_text(f"{pid} (python (fit) worker) {state} 1 2 3\n", encoding="ascii")


def test_a_zombie_parent_counts_as_dead(tmp_path, monkeypatch):
    # Codex review on #30: os.kill(pid, 0) succeeds on a zombie, so the
    # watchdog would keep a CPU-bound worker alive after the runner died.
    monkeypatch.setattr(_worker_lifetime.os, "kill", lambda pid, sig: None)
    _proc(tmp_path, 4242, "Z")
    assert _worker_lifetime._posix_alive(4242, proc_root=str(tmp_path)) is False


def test_running_parent_alive_and_missing_proc_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(_worker_lifetime.os, "kill", lambda pid, sig: None)
    _proc(tmp_path, 4242, "S")
    assert _worker_lifetime._posix_alive(4242, proc_root=str(tmp_path)) is True
    assert _worker_lifetime._posix_alive(5151, proc_root=str(tmp_path)) is True


def test_a_vanished_process_is_dead(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(_worker_lifetime.os, "kill", gone)
    assert _worker_lifetime._posix_alive(os.getpid()) is False
