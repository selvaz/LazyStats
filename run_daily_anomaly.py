# -*- coding: utf-8 -*-
"""Daily anomaly investigation: gate, and optionally explain.

Two plans that share a gate and nothing else.

``gate-shadow`` loads an already-captured input artifact, evaluates the gate
and writes one canonical JSON artifact. It does not import, construct or
call a language model, a browser, a database, a report renderer or a
messenger — not disabled versions of them, absent. That is what makes a
shadow run comparable byte for byte against a live one, and what makes it
free to run every day.

``live-explain`` is the production path: gate, explain, persist, render,
send. Its dependencies are passed in explicitly rather than imported at
module level, so the shadow plan cannot reach them through this module.

The preset — thresholds, the upstream series, which model explains — is not
here. See :mod:`lazystats.anomaly_gate_config`; ``--config`` is required and
there is no default, because a gate silently running at someone else's
sensitivity either floods the investigation or misses the day that mattered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lazystats.anomaly_gate import evaluate_gate
from lazystats.anomaly_gate_config import AnomalyGateConfig, GateConfigError, load_gate_config

#: Fixed timestamp in shadow artifacts: they are compared byte for byte
#: against each other, and a wall-clock field would differ every run for
#: reasons unrelated to the gate.
SHADOW_GENERATED_AT = "1970-01-01T00:00:00+00:00"

#: Steps that must never appear in a shadow plan.
LIVE_ONLY_STEPS = frozenset({"explain", "persist", "render", "send"})


class RunError(RuntimeError):
    """The run cannot proceed. Distinguished from an analysis failure."""


@dataclass(frozen=True)
class RunContext:
    """Everything a run needs, stated rather than discovered.

    No name, path or identifier from any particular project appears in this
    package: they all arrive here.
    """

    config: AnomalyGateConfig
    upstream_series_key: str
    upstream_produced_by: str
    input_artifact: Path
    output_dir: Path
    mode: str  # "gate-shadow" | "live-explain"


def load_input_artifact(path: Path) -> dict:
    """Read a captured input: the two consecutive payloads and their ids.

    Reading a captured artifact rather than a database is what keeps the
    shadow plan free of infrastructure — and it means live and shadow can be
    driven from exactly the same input, which is the only way their outputs
    are comparable.
    """
    if not path.is_file():
        raise RunError(f"input artifact not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunError(f"input artifact is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RunError(f"input artifact unreadable: {exc}") from exc

    for key in ("current", "previous", "trigger_result_id"):
        if key not in data:
            raise RunError(f"input artifact is missing '{key}'")
    return data


def gate_step(ctx: RunContext) -> dict:
    """Evaluate the gate. Pure with respect to the outside world."""
    data = load_input_artifact(ctx.input_artifact)
    already = frozenset(
        (i["instrument"], i["date"]) for i in data.get("already_investigated", [])
    )
    targets = evaluate_gate(
        current=data["current"],
        previous=data["previous"],
        trigger_result_id=data["trigger_result_id"],
        config=ctx.config,
        already_investigated=already,
    )
    return {
        "schema_version": "1.0",
        "generated_at": SHADOW_GENERATED_AT,
        "upstream_series_key": ctx.upstream_series_key,
        "upstream_produced_by": ctx.upstream_produced_by,
        "trigger_result_id": data["trigger_result_id"],
        "as_of": data["current"]["as_of"],
        "gate_parameters": ctx.config.as_provenance(),
        "targets": [t.as_dict() for t in targets],
        "anomaly_count": sum(len(t.items) for t in targets),
    }


def write_gate_artifact(ctx: RunContext, artifact: dict) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.output_dir / f"anomaly_gate_{artifact['as_of']}.json"
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True), encoding="utf-8")
    return out


def build_shadow_plan(ctx: RunContext) -> list[tuple[str, Callable[[], Any]]]:
    """Gate, then write. Nothing else exists in this plan."""
    state: dict[str, Any] = {}

    def gate() -> dict:
        state["artifact"] = gate_step(ctx)
        return state["artifact"]

    def write() -> dict:
        path = write_gate_artifact(ctx, state["artifact"])
        return {"artifact_path": str(path),
                "anomaly_count": state["artifact"]["anomaly_count"]}

    return [("gate", gate), ("write_gate_artifact", write)]


def build_live_plan(
    ctx: RunContext,
    *,
    explain: Callable[[dict], Any],
    persist: Callable[[dict], Any],
    render: Callable[[dict], Any],
    send: Callable[[dict], Any],
) -> list[tuple[str, Callable[[], Any]]]:
    """The production path.

    Every side-effecting stage is injected. This module therefore imports no
    model, database, renderer or messenger, and a shadow run cannot reach
    one through it — which is checked by loading this module in a fresh
    process and inspecting ``sys.modules``.
    """
    state: dict[str, Any] = {}

    def gate() -> dict:
        state["artifact"] = gate_step(ctx)
        return state["artifact"]

    return [
        ("gate", gate),
        ("explain", lambda: state.setdefault("explained", explain(state["artifact"]))),
        ("persist", lambda: persist(state["explained"])),
        ("render", lambda: render(state["explained"])),
        ("send", lambda: send(state["explained"])),
    ]


def is_inside(candidate: str | Path, protected: str | Path) -> bool:
    """Whether ``candidate`` is ``protected`` or sits under it.

    Canonical paths, not strings: on Windows a different case names the same
    directory, and a junction reaches the same tree under another name.
    """
    prot = os.path.normcase(os.path.realpath(protected))
    targ = os.path.normcase(os.path.realpath(candidate))
    if targ == prot:
        return True
    try:
        return os.path.commonpath([targ, prot]) == prot
    except ValueError:
        return False  # different drives cannot contain one another


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, metavar="PATH",
                    help="Gate configuration (TOML). Required: there is no default preset.")
    ap.add_argument("--input-artifact", required=True, metavar="PATH",
                    help="Captured input: current and previous payloads plus the trigger id.")
    ap.add_argument("--output-dir", required=True, metavar="PATH",
                    help="Isolated directory for the gate artifact.")
    ap.add_argument("--upstream-series-key", required=True,
                    help="Series the upstream job writes. Stated, never assumed.")
    ap.add_argument("--upstream-produced-by", default=None,
                    help="Producer identity; defaults to the configuration's value.")
    ap.add_argument("--protected-dir", default=None, metavar="PATH",
                    help="A directory the run must not write into, such as a production "
                         "reports tree. Refused along with anything inside it.")
    ap.add_argument("--mode", choices=["gate-shadow"], default="gate-shadow",
                    help="Only gate-shadow runs from this entry point. The live path is "
                         "composed by its caller, which supplies the explaining, "
                         "persisting, rendering and sending stages explicitly.")
    args = ap.parse_args(argv)

    try:
        config = load_gate_config(args.config)
    except GateConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    if args.protected_dir and is_inside(out_dir, args.protected_dir):
        print(f"CONFIG ERROR: --output-dir must not be inside {args.protected_dir}",
              file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"CONFIG ERROR: --output-dir must be new or empty, and {out_dir} is not",
              file=sys.stderr)
        return 2

    ctx = RunContext(
        config=config,
        upstream_series_key=args.upstream_series_key,
        upstream_produced_by=args.upstream_produced_by or config.upstream_produced_by,
        input_artifact=Path(args.input_artifact),
        output_dir=out_dir,
        mode=args.mode,
    )

    try:
        result: Any = None
        for _name, step in build_shadow_plan(ctx):
            result = step()
    except RunError as exc:
        print(f"RUN ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
