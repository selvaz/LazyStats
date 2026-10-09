"""Killable, spawn-safe isolation for each daily symbol fit."""
from __future__ import annotations

import json
import math
import multiprocessing
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from lazystats._fit_worker import fit_worker

DEFAULT_FIT_TIMEOUT_SECONDS = 300.0


class FitTimeout(TimeoutError):
    """The child exceeded the symbol's wall-clock budget and was reaped."""


class FitProcessError(RuntimeError):
    """The child raised or exited without a complete fit."""


def run_fit_in_process(instrument: str, *, timeout_seconds: float = DEFAULT_FIT_TIMEOUT_SECONDS,
                       fit_function: Callable | None = None, **kwargs) -> dict:
    """Fit in a fresh child, terminate on deadline, and always reap it.

    The budget includes spawn, data loading, fitting and chart generation.
    Results use a private scratch file: joining before draining a Queue/Pipe
    can deadlock when the chart or historical readings exceed its buffer.
    Only the parent persists results; the child never receives a depot.
    ``fit_function`` permits spawn-importable doubles in offline tests.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("fit timeout must be finite and positive")
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="lazystats-fit-") as scratch:
        result_path = Path(scratch) / "result.json"
        process = context.Process(target=fit_worker,
                                  args=(str(result_path), instrument, kwargs, fit_function))
        started = time.monotonic()
        try:
            process.start()
            process.join(max(0.0, timeout_seconds - (time.monotonic() - started)))
            if process.is_alive():
                raise FitTimeout(f"{instrument}: fit exceeded {timeout_seconds:g}s")
            if process.exitcode != 0 or not result_path.is_file():
                raise FitProcessError(f"{instrument}: fit child exited with code {process.exitcode}")
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if "error" in record:
                raise FitProcessError(record["error"])
            return record["result"]
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                    if process.is_alive():
                        process.kill()
                process.join()
            process.close()
