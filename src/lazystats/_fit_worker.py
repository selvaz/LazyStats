"""Minimal spawn entry point; scientific imports happen inside the fit budget."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def fit_worker(result_path: str, instrument: str, kwargs: dict[str, Any],
               fit_function: Callable[..., dict[str, Any]] | None) -> None:
    record: dict[str, Any]
    try:
        if fit_function is None:
            from lazystats.regimes.estimation import fit_symbol

            fit_function = fit_symbol
        record = {"result": fit_function(instrument, **kwargs)}
    except Exception:
        record = {"error": traceback.format_exc()}
    path = Path(result_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _resolve_function(reference: dict[str, Any] | None) -> Callable[..., dict[str, Any]] | None:
    if reference is None:
        return None
    name = reference["module"]
    if name in ("__main__", "__mp_main__"):
        spec = importlib.util.spec_from_file_location("_lazystats_fit_callable", reference["file"])
        if spec is None or spec.loader is None:
            raise ImportError("cannot load the fit callable's source file")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(name)
    function: Any = module
    for part in reference["name"].split("."):
        function = getattr(function, part)
    return function  # type: ignore[no-any-return]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--watch-worker", type=int)
    args = parser.parse_args()
    if args.watch_worker is not None:
        from lazystats._worker_lifetime import watch_parent

        watch_parent(args.parent_pid, args.watch_worker)
        return 0

    handshake = sys.stdin.readline()
    if not handshake:
        return 0
    protected = json.loads(handshake)["job_protected"]
    guardian = None
    if not protected:
        guardian = subprocess.Popen(
            [sys.executable, "-m", "lazystats._fit_worker", "--parent-pid", str(args.parent_pid),
             "--watch-worker", str(os.getpid())], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        # Keep stdin reads on this thread: a concurrent blocking CRT read can
        # stall native scientific-library initialization on Windows. The job
        # object or independent guardian handles parent death during a fit.
        for line in sys.stdin:
            request = json.loads(Path(json.loads(line)).read_text(encoding="utf-8"))
            try:
                function = _resolve_function(request["fit_function"])
                fit_worker(request["result_path"], request["instrument"],
                           request["kwargs"], function)
            except Exception:
                path = Path(request["result_path"])
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"error": traceback.format_exc()}),
                                     encoding="utf-8")
                temporary.replace(path)
    finally:
        if guardian is not None:
            guardian.terminate()
            guardian.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
