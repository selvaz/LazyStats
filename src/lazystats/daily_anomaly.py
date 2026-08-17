"""Daily anomaly investigation: two plans that share a gate and nothing else.

``build_shadow_plan`` evaluates the gate on an already-captured input and
writes one canonical JSON artifact. It does not import, construct or call a
language model, a browser, a database, a report renderer or a messenger —
not disabled versions of them, absent. That is what makes a shadow run
comparable against a live one, and what makes it free to run every day.

``build_live_plan`` is the production path: gate, explain, persist, render,
send. Every side-effecting stage is passed in by its caller rather than
imported here, so nothing in this module can reach one.

Everything else is refusal. A shadow run exists to be harmless, and a
harmless run is one that cannot write where it should not: the output
directory must be stated, must be new or empty, and must sit outside every
directory the caller declares protected — including after the artifact's own
filename is joined onto it, since that name comes from the input file. The
date in that name is validated as a plain ISO date before it is ever used as
a path component.

The preset — thresholds, the upstream series, which model explains — is not
here. See :mod:`lazystats.anomaly_gate_config`.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lazystats.anomaly_gate import evaluate_gate
from lazystats.anomaly_gate_config import AnomalyGateConfig

#: Fixed timestamp in shadow artifacts: they are compared byte for byte
#: against each other, and a wall-clock field would differ every run for
#: reasons that have nothing to do with the gate.
SHADOW_GENERATED_AT = "1970-01-01T00:00:00+00:00"

#: Steps that must never appear in a shadow plan.
LIVE_ONLY_STEPS = frozenset({"explain", "persist", "render", "send"})

#: A plain ISO calendar date and nothing else. The value becomes part of a
#: filename, so anything carrying a separator, a drive letter or a parent
#: reference has to be refused before it is joined onto a directory.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RunError(RuntimeError):
    """The run cannot proceed.

    Distinct from an analysis result: this says the job could not be run as
    asked, not that it ran and found nothing.
    """


class SetupError(RunError):
    """The run was asked for wrongly, and nothing has been attempted yet.

    Separate from its parent so a caller can tell "this invocation is
    malformed" from "the job started and could not finish". Everything
    refused before the plan is built is one of these.
    """


@dataclass(frozen=True)
class RunContext:
    """Everything a run needs, stated rather than discovered.

    No name, path or identifier from any particular project appears in this
    package: the series identity comes from ``config``, and the paths come
    from the caller.

    There is no ``mode`` field. Which plan is running is expressed by which
    builder was called; a mode string alongside them would be a second
    statement of the same fact, free to contradict the first.
    """

    config: AnomalyGateConfig
    input_artifact: Path
    output_dir: Path
    protected_dirs: tuple[Path, ...]


def validate_as_of(value: Any) -> str:
    """Accept a real calendar date in plain ISO form; refuse anything else.

    Two checks, because they catch different things. The pattern refuses
    whatever could steer a path — a separator, a drive letter, a parent
    reference, an embedded NUL — before the value is ever joined onto a
    directory. Parsing then refuses dates that look right and are not:
    2026-02-30, 2026-13-01 and 2026-00-10 all pass any shape test while
    naming no day.
    """
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        raise RunError(
            f"'as_of' must be an ISO date such as 2026-08-10, got {value!r}"
        )
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise RunError(f"'as_of' is not a real calendar date: {value!r} ({exc})") from exc
    return value


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


def assert_outside_protected(path: str | Path, protected: tuple[Path, ...]) -> None:
    """Refuse a path that lands in, or under, any protected directory.

    Applied to the output directory *and* to the artifact's final path. The
    filename carries a date read from the input file, so checking only the
    directory would trust a value the run did not choose.
    """
    if not protected:
        raise SetupError("at least one protected directory must be declared")
    for prot in protected:
        if is_inside(path, prot):
            raise SetupError(f"{path} is inside the protected directory {prot}")


def prepare_output_dir(ctx: RunContext) -> None:
    """Check every precondition on where this run may write.

    Called once, from :func:`run_shadow`, before the plan is built — so the
    guarantee holds for anyone who runs a shadow plan, not only for callers
    who remembered to check first. The command-line wrapper repeats none of
    it.

    A protected directory that does not exist is refused rather than
    created. The argument names a tree to stay out of; if it is absent, the
    likeliest reason is a typo, and a typo here protects nothing while
    looking exactly like protection.
    """
    if not ctx.protected_dirs:
        raise SetupError("at least one protected directory must be declared")

    for prot in ctx.protected_dirs:
        try:
            if not prot.exists():
                raise SetupError(f"protected directory does not exist: {prot}")
            if not prot.is_dir():
                raise SetupError(f"protected path is not a directory: {prot}")
        except OSError as exc:
            raise SetupError(f"protected directory unusable: {prot} ({exc})") from exc

    assert_outside_protected(ctx.output_dir, ctx.protected_dirs)

    try:
        if ctx.output_dir.exists():
            if not ctx.output_dir.is_dir():
                raise SetupError(f"output directory is not a directory: {ctx.output_dir}")
            if any(ctx.output_dir.iterdir()):
                raise SetupError(
                    f"output directory must be new or empty, and {ctx.output_dir} is not"
                )
    except OSError as exc:
        raise SetupError(f"output directory unusable: {ctx.output_dir} ({exc})") from exc


def load_input_artifact(path: Path) -> dict[str, Any]:
    """Read a captured input: two consecutive payloads and their trigger id.

    Reading a captured artifact rather than a database is what keeps the
    shadow plan free of infrastructure, and it means the live and shadow
    paths can be driven from exactly the same input — the only way their
    outputs are comparable.

    Every field is checked here rather than where it is used. A payload that
    is a list, or a trigger id that is blank, would otherwise surface deep
    inside the gate as a ``TypeError`` or as a result quietly attributed to
    nothing.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RunError(f"input artifact not found: {path}") from exc
    except OSError as exc:
        raise RunError(f"input artifact unreadable: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunError(f"input artifact is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise RunError(
            f"input artifact must be a JSON object, got {type(data).__name__}"
        )

    for key in ("current", "previous"):
        if key not in data:
            raise RunError(f"input artifact is missing '{key}'")
        if not isinstance(data[key], dict):
            raise RunError(
                f"input artifact's '{key}' must be an object, "
                f"got {type(data[key]).__name__}"
            )

    trigger = data.get("trigger_result_id")
    if not isinstance(trigger, str) or not trigger.strip():
        raise RunError("input artifact's 'trigger_result_id' must be a non-empty string")

    if "as_of" not in data["current"]:
        raise RunError("input artifact's 'current' payload is missing 'as_of'")
    current_as_of = validate_as_of(data["current"]["as_of"])

    investigated = data.get("already_investigated", [])
    if not isinstance(investigated, list):
        raise RunError(
            f"'already_investigated' must be a list, got {type(investigated).__name__}"
        )
    for i, entry in enumerate(investigated):
        where = f"already_investigated[{i}]"
        if not isinstance(entry, dict):
            raise RunError(f"{where} must be an object, got {type(entry).__name__}")
        for key in ("instrument", "date"):
            if key not in entry:
                raise RunError(f"{where} is missing '{key}'")
        instrument = entry["instrument"]
        # Not trimmed, not coerced: this set is matched by exact equality, so
        # " ticker:AAA" quietly becoming "ticker:AAA" would decide which
        # anomalies get suppressed on the strength of a typo.
        if (not isinstance(instrument, str) or not instrument
                or instrument.strip() != instrument):
            raise RunError(
                f"{where}: 'instrument' must be a non-empty string without "
                f"surrounding whitespace, got {instrument!r}"
            )
        try:
            validate_as_of(entry["date"])
        except RunError as exc:
            raise RunError(f"{where}: {exc}") from exc

    for key in ("upstream_series_key", "upstream_produced_by"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise RunError(
                f"input artifact's '{key}' must be a non-empty string without "
                f"surrounding whitespace"
            )

    if "as_of" not in data["previous"]:
        raise RunError("input artifact's 'previous' payload is missing 'as_of'")
    previous_as_of = validate_as_of(data["previous"]["as_of"])
    if previous_as_of >= current_as_of:
        raise RunError(
            "input artifact's 'previous.as_of' must predate 'current.as_of'; "
            f"got previous={previous_as_of}, current={current_as_of}"
        )

    for label in ("current", "previous"):
        payload = data[label]
        nested_objects = {
            "outliers_last5": "outliers",
            "volatility_short": "volatility",
            "volatility_long": "volatility",
            "correlation_short": "correlation",
        }
        for block_name, value_name in nested_objects.items():
            block = payload.get(block_name)
            if not isinstance(block, dict):
                raise RunError(
                    f"input artifact's '{label}.{block_name}' must be an object"
                )
            value = block.get(value_name)
            expected = list if value_name == "outliers" else dict
            if not isinstance(value, expected):
                raise RunError(
                    f"input artifact's '{label}.{block_name}.{value_name}' "
                    f"must be a {expected.__name__}"
                )
            # The shape check above stops at the outer container: it would
            # accept {"ticker:A": "STRINGA"} as a valid volatility block. The
            # gate then does `s.get("annualized_volatility") if s else None`
            # on every inner value, which raises AttributeError on a non-empty
            # string rather than treating it as the malformed data it is.
            where = f"{label}.{block_name}.{value_name}"
            if value_name == "outliers":
                for i, entry in enumerate(value):
                    if not isinstance(entry, dict):
                        raise RunError(f"{where}[{i}] must be an object, got {type(entry).__name__}")
            else:
                for inner_key, entry in value.items():
                    if entry is not None and not isinstance(entry, dict):
                        raise RunError(
                            f"{where}[{inner_key!r}] must be an object or null, got {type(entry).__name__}"
                        )
        if not isinstance(payload.get("returns_table"), dict):
            raise RunError(f"input artifact's '{label}.returns_table' must be an object")

    comparison = data.get("comparison")
    if not isinstance(comparison, dict):
        raise RunError("input artifact's 'comparison' must be an object")
    if comparison.get("selection_policy") != "latest_two_stable_rows":
        raise RunError(
            "input artifact's 'comparison.selection_policy' must be "
            "'latest_two_stable_rows'"
        )
    for key in ("current_result_id", "previous_result_id"):
        value = comparison.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise RunError(
                f"input artifact's 'comparison.{key}' must be a non-empty string "
                "without surrounding whitespace"
            )
    if comparison["current_result_id"] != trigger:
        raise RunError(
            "input artifact's 'comparison.current_result_id' must match "
            "'trigger_result_id'"
        )
    if comparison["previous_result_id"] == comparison["current_result_id"]:
        raise RunError("comparison current and previous result ids must differ")
    if data.get("schema_version") != "1.1":
        raise RunError("input artifact's 'schema_version' must be '1.1'")
    return data


def _validate_input_identity(data: dict[str, Any], config: AnomalyGateConfig) -> None:
    expected = {
        "upstream_series_key": config.upstream_series_key,
        "upstream_produced_by": config.upstream_produced_by,
    }
    for key, configured in expected.items():
        captured = data[key]
        if captured != configured:
            raise RunError(
                f"input artifact's '{key}' does not match the configured identity"
            )


def gate_step(ctx: RunContext) -> dict[str, Any]:
    """Evaluate the gate. Pure with respect to the outside world."""
    data = load_input_artifact(ctx.input_artifact)
    _validate_input_identity(data, ctx.config)
    already = frozenset(
        (entry["instrument"], entry["date"]) for entry in data.get("already_investigated", [])
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
        "upstream_series_key": ctx.config.upstream_series_key,
        "upstream_produced_by": ctx.config.upstream_produced_by,
        "trigger_result_id": data["trigger_result_id"],
        "as_of": data["current"]["as_of"],
        "gate_parameters": ctx.config.as_provenance(),
        "targets": [t.as_dict() for t in targets],
        "anomaly_count": sum(len(t.items) for t in targets),
    }


def artifact_path(ctx: RunContext, as_of: str) -> Path:
    """Where the gate artifact goes, checked before anything is written."""
    out = ctx.output_dir / f"anomaly_gate_{validate_as_of(as_of)}.json"
    assert_outside_protected(ctx.output_dir, ctx.protected_dirs)
    # The parent of the resolved path, not the declared output directory: a
    # filename is not supposed to be able to climb out, and this is where
    # that assumption is checked rather than assumed.
    if Path(os.path.realpath(out)).parent != Path(os.path.realpath(ctx.output_dir)):
        raise RunError(f"the artifact name would write outside {ctx.output_dir}")
    return out


def write_gate_artifact(ctx: RunContext, artifact: dict[str, Any]) -> Path:
    out = artifact_path(ctx, artifact["as_of"])
    try:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=1, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        raise RunError(f"could not write the gate artifact: {exc}") from exc
    return out


def build_shadow_plan(ctx: RunContext) -> list[tuple[str, Callable[[], Any]]]:
    """Gate, then write. Nothing else exists in this plan."""
    state: dict[str, Any] = {}

    def gate() -> dict[str, Any]:
        artifact = gate_step(ctx)
        state["artifact"] = artifact
        return artifact

    def write() -> dict[str, Any]:
        path = write_gate_artifact(ctx, state["artifact"])
        return {"artifact_path": str(path),
                "anomaly_count": state["artifact"]["anomaly_count"]}

    return [("gate", gate), ("write_gate_artifact", write)]


def build_live_plan(
    ctx: RunContext,
    *,
    explain: Callable[[dict[str, Any]], Any],
    persist: Callable[[Any], Any],
    render: Callable[[Any], Any],
    send: Callable[[Any], Any],
) -> list[tuple[str, Callable[[], Any]]]:
    """The production path.

    Every side-effecting stage is injected. This module therefore imports no
    model, database, renderer or messenger, and a shadow run cannot reach one
    through it — which is checked by loading it in a fresh process and
    inspecting ``sys.modules``.
    """
    state: dict[str, Any] = {}

    def gate() -> dict[str, Any]:
        artifact = gate_step(ctx)
        state["artifact"] = artifact
        return artifact

    return [
        ("gate", gate),
        ("explain", lambda: state.setdefault("explained", explain(state["artifact"]))),
        ("persist", lambda: persist(state["explained"])),
        ("render", lambda: render(state["explained"])),
        ("send", lambda: send(state["explained"])),
    ]


def run_shadow(ctx: RunContext) -> dict[str, Any]:
    """Check the preconditions, then execute the shadow plan.

    The checks are here rather than in the caller so that every route into a
    shadow run passes through them.
    """
    prepare_output_dir(ctx)
    result: dict[str, Any] = {}
    for _name, step in build_shadow_plan(ctx):
        result = step()
    return result


__all__ = [
    "LIVE_ONLY_STEPS",
    "SHADOW_GENERATED_AT",
    "RunContext",
    "RunError",
    "SetupError",
    "artifact_path",
    "assert_outside_protected",
    "build_live_plan",
    "build_shadow_plan",
    "gate_step",
    "is_inside",
    "load_input_artifact",
    "prepare_output_dir",
    "run_shadow",
    "validate_as_of",
    "write_gate_artifact",
]
