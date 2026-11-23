"""Reusable, killable child-process isolation for daily symbol fits."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from lazystats._worker_lifetime import kill_on_close_job

DEFAULT_FIT_TIMEOUT_SECONDS = 300.0


class FitTimeout(TimeoutError):
    """The child exceeded a request's wall-clock budget and was reaped."""


class FitProcessError(RuntimeError):
    """The child raised or exited without a complete fit."""


class FitWorker:
    """One worker per daily loop; replace it only after timeout or crash.

    Popen with -m avoids multiprocessing's __mp_main__ replay. Each deadline
    includes first-worker startup, loading, fitting and charts. Stdin carries
    short request-file paths so large requests/results cannot block a pipe.
    Fits are sequential; only the parent writes the result depot.
    """

    def __init__(self, *, use_job_object: bool = True):
        self.process: subprocess.Popen | None = None
        self._job = None
        self._use_job_object = use_job_object
        self._scratch = TemporaryDirectory(prefix="lazystats-fit-")
        self._sequence = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                return
            self._stop()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[2]), os.getcwd(),
             *(str(p) for p in sys.path if p)])
        self.process = subprocess.Popen(
            [sys.executable, "-m", "lazystats._fit_worker", "--parent-pid", str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, env=env,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if self._use_job_object:
            self._job = kill_on_close_job(self.process.pid)
        self.process.stdin.write(json.dumps({"job_protected": self._job is not None}) + "\n")
        self.process.stdin.flush()

    def _stop(self) -> None:
        started = time.monotonic()
        process, self.process = self.process, None
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    # Do not let cleanup turn a bounded fit into an unbounded
                    # job.  A second kill is harmless; make one final bounded
                    # reap attempt rather than trusting an unbounded wait.
                    process.kill()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                if process.stdin is not None:
                    process.stdin.close()
        finally:
            if self._job is not None:
                self._job.close()
                self._job = None
            print(f"regime worker phase=stop seconds={time.monotonic() - started:.3f}",
                  file=sys.stderr, flush=True)

    def close(self) -> None:
        self._stop()
        self._scratch.cleanup()
        self._closed = True

    def reset(self) -> None:
        """Reap the current child; the next request starts a clean worker."""
        if self._closed:
            raise RuntimeError("fit worker is closed")
        self._stop()

    def fit(self, instrument: str, *, timeout_seconds: float = DEFAULT_FIT_TIMEOUT_SECONDS,
            fit_function: Callable | None = None, **kwargs) -> dict:
        if self._closed:
            raise RuntimeError("fit worker is closed")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("fit timeout must be finite and positive")
        started = time.monotonic()
        polls = 0
        self._sequence += 1
        request_path = Path(self._scratch.name) / f"request-{self._sequence}.json"
        result_path = Path(self._scratch.name) / f"result-{self._sequence}.json"
        reference = None
        if fit_function is not None:
            module = sys.modules[fit_function.__module__]
            reference = {"module": fit_function.__module__, "name": fit_function.__qualname__,
                         "file": getattr(module, "__file__", None)}
        request_path.write_text(json.dumps({"instrument": instrument, "kwargs": kwargs,
                                           "fit_function": reference,
                                           "result_path": str(result_path)}), encoding="utf-8")
        try:
            self._start()
            self.process.stdin.write(json.dumps(str(request_path)) + "\n")
            self.process.stdin.flush()
            while True:
                polls += 1
                if time.monotonic() - started >= timeout_seconds:
                    self._stop()
                    print(f"regime worker phase=poll polls={polls} "
                          f"seconds={time.monotonic() - started:.3f} status=timeout",
                          file=sys.stderr, flush=True)
                    raise FitTimeout(f"{instrument}: fit exceeded {timeout_seconds:g}s")
                if result_path.is_file():
                    try:
                        record = json.loads(result_path.read_text(encoding="utf-8"))
                    except PermissionError:
                        # Windows can deny the open for a moment right after the child's
                        # atomic replace (30/10 Regime_8y); retry within the fit timeout.
                        time.sleep(0.05)
                        continue
                    if "error" in record:
                        raise FitProcessError(record["error"])
                    print(f"regime worker phase=poll polls={polls} "
                          f"seconds={time.monotonic() - started:.3f} status=ok",
                          file=sys.stderr, flush=True)
                    return record["result"]
                code = self.process.poll()
                if code is not None:
                    self._stop()
                    raise FitProcessError(f"{instrument}: fit child exited with code {code}")
                time.sleep(min(0.01, max(0, timeout_seconds - (time.monotonic() - started))))
        except FitTimeout:
            raise
        except (BrokenPipeError, OSError) as exc:
            self._stop()
            raise FitProcessError(f"{instrument}: worker transport failed: {exc}") from exc
        except BaseException:
            if sys.exc_info()[0] not in (FitProcessError, FitTimeout):
                self._stop()
            raise
        finally:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            result_path.with_suffix(".tmp").unlink(missing_ok=True)


def run_fit_in_process(instrument: str, *, timeout_seconds: float = DEFAULT_FIT_TIMEOUT_SECONDS,
                       fit_function: Callable | None = None, **kwargs) -> dict:
    """One-shot compatibility API; batch callers should reuse a FitWorker."""
    with FitWorker() as worker:
        return worker.fit(instrument, timeout_seconds=timeout_seconds,
                          fit_function=fit_function, **kwargs)
