"""Command-line entry point for one shadow run of the daily anomaly gate.

A thin wrapper. Everything it does beyond parsing arguments lives in
:mod:`lazystats.daily_anomaly`, which is an ordinary importable module — so
the behaviour is tested by importing the package, not by loading this file
by path.

Only the shadow path runs from here. The live path composes explaining,
persisting, rendering and sending, and those stages are supplied by whoever
runs it rather than chosen here.

The series identity is not an option. It comes from the configuration, so a
run cannot claim to be using one preset while reading a different series.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lazystats.anomaly_gate_config import GateConfigError, load_gate_config
from lazystats.daily_anomaly import RunContext, RunError, SetupError, run_shadow


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, metavar="PATH",
                    help="Gate configuration (TOML). Required: there is no default "
                         "preset, and it declares the upstream series.")
    ap.add_argument("--input-artifact", required=True, metavar="PATH",
                    help="Captured input: current and previous payloads plus the "
                         "trigger id.")
    ap.add_argument("--output-dir", required=True, metavar="PATH",
                    help="Isolated directory for the gate artifact. Must be new or "
                         "empty.")
    ap.add_argument("--protected-dir", required=True, action="append", metavar="PATH",
                    dest="protected_dirs",
                    help="A directory this run must not write into, such as a "
                         "production reports tree. Required, and repeatable. "
                         "Refused along with anything inside it.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_gate_config(args.config)
    except GateConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    ctx = RunContext(
        config=config,
        input_artifact=Path(args.input_artifact),
        output_dir=Path(args.output_dir),
        protected_dirs=tuple(Path(p) for p in args.protected_dirs),
    )

    # The preconditions on where a run may write are checked by run_shadow,
    # not here. Repeating them would put the guarantee in two places, free to
    # drift apart, and would leave any other caller of the plan unprotected.
    # This only maps the two failures onto different exit codes: a malformed
    # invocation is not the same as a job that started and could not finish.
    try:
        result = run_shadow(ctx)
    except SetupError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except RunError as exc:
        print(f"RUN ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
