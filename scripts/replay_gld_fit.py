#!/usr/bin/env python
"""Manual GLD weekend diagnosis; never run in CI or write to the source HUB.

From this worktree:
    C:/ProgramData/spyder-6/python.exe scripts/replay_gld_fit.py --hub PATH

Copies HUB into a TemporaryDirectory only when C: will retain >=3 GiB free.
Extracts the same daily log returns as the runner, up to --as-of, through the
hub's read-only reader. Both fits use identical parameters and fresh spawned
processes. The second fit drops Sat/Sun *return rows*, without recomputing
weekday returns. No charts, depot writes, delivery, or network calls.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# A diagnostic launched by filename must replay this checkout, even when a
# different worktree is installed editable in the selected interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lazystats.regimes.fit_process import FitTimeout, run_fit_in_process  # noqa: E402


def replay(symbol: str, *, dates: list[str], values: list[float], s_max: int,
           n_starts: int, random_state: int) -> dict:
    import pandas as pd

    from lazystats.regimes import MSRegimeEngine

    frame = pd.DataFrame({symbol: values}, index=pd.to_datetime(dates))
    engine = MSRegimeEngine(S_max=s_max, n_starts=n_starts, random_state=random_state)
    started = time.perf_counter()
    cpu_started = time.process_time()
    run = engine.fit(frame)
    return {"fit_seconds": round(time.perf_counter() - started, 6),
            "cpu_seconds": round(time.process_time() - cpu_started, 6),
            "n_states": int(run.meta[symbol]["S"]), "n_obs": len(frame),
            "bic": float(run.meta[symbol]["bic"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", required=True, type=Path)
    parser.add_argument("--as-of", default="2026-10-01")
    parser.add_argument("--start", default="")
    parser.add_argument("--s-max", type=int, default=3)
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    size = args.hub.stat().st_size
    free = shutil.disk_usage("C:/" if sys.platform == "win32" else args.hub.anchor).free
    minimum = 3 * 1024**3
    print(json.dumps({"hub_bytes": size, "free_bytes_before": free,
                      "projected_free_bytes_after": free - size,
                      "minimum_free_bytes": minimum}), flush=True)
    if free - size < minimum:
        print("SKIPPED: copying HUB would leave C: with less than 3 GiB free.", flush=True)
        return 3

    from datetime import date

    from market_data_hub.extract import extract_returns

    with tempfile.TemporaryDirectory(prefix="lazystats-gld-replay-") as scratch:
        copied = Path(scratch) / args.hub.name
        shutil.copy2(args.hub, copied)
        print(json.dumps({"scratch_copy": str(copied), "as_of": args.as_of,
                          "start": args.start or "full history", "s_max": args.s_max,
                          "n_starts": args.n_starts, "random_state": args.random_state,
                          "free_bytes_after_copy": shutil.disk_usage(copied.parent).free}),
              flush=True)
        frame, metadata = extract_returns(["GLD"], start=args.start or None,
                                          end=args.as_of, frequency="D", db_path=str(copied))
        series = frame["GLD"].dropna()
        dates = [str(d.date()) for d in series.index]
        values = [float(v) for v in series]
        weekday = [date.fromisoformat(d).weekday() < 5 for d in dates]
        print(json.dumps({"n_returns": len(dates), "weekend_returns": weekday.count(False),
                          "data_start": dates[0], "data_end": dates[-1],
                          "used_returns_view": metadata.get("used_returns_view")}), flush=True)
        results = {}
        for label, keep in (("with_weekends", [True] * len(dates)), ("without_weekends", weekday)):
            print(f"Starting {label}", flush=True)
            started = time.perf_counter()
            try:
                result = run_fit_in_process(
                    "GLD", timeout_seconds=args.timeout_seconds, fit_function=replay,
                    dates=[d for d, k in zip(dates, keep, strict=True) if k],
                    values=[v for v, k in zip(values, keep, strict=True) if k],
                    s_max=args.s_max, n_starts=args.n_starts, random_state=args.random_state)
            except FitTimeout as exc:
                result = {"status": "timeout", "detail": str(exc), "n_states": None}
            result["process_seconds"] = round(time.perf_counter() - started, 6)
            results[label] = result
            print(json.dumps({label: result}), flush=True)
        full = results["with_weekends"].get("fit_seconds")
        clean = results["without_weekends"].get("fit_seconds")
        if full is not None and clean is not None:
            ratio = full / clean
            print(f"Fit speedup after removing weekends: {ratio:.3f}x", flush=True)
            if weekday.count(False) and ratio >= 1.25:
                print(f"FINDING: removing weekend rows reduced fit wall time by "
                      f"{100 * (1 - clean / full):.1f}% (>=25% speedup); "
                      "this supports W0-08 as a contributor to fit slowness. "
                      "A single replay on a shared host does not establish the sole cause.",
                      flush=True)
            else:
                print("FINDING: this replay does not establish weekend rows as a cause "
                      "of substantial GLD fit slowness (25% speedup threshold).", flush=True)
        else:
            print("INCONCLUSIVE: a replay timed out; no completed state count for that fit.",
                  flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
