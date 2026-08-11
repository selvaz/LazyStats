# -*- coding: utf-8 -*-
"""The refactored gate must select exactly what the legacy one selected.

The legacy gate is executed, not transcribed. It is imported unmodified from
the production checkout and driven through its own entry point,
``find_investigation_targets(depot_path=..., explanations_depot_path=...)``,
against SQLite depots this test builds in a temporary directory. No line of
its source is rewritten, nothing is monkeypatched, nothing is put on
sys.path, and none of its logic is restated here — the comparison is between
two programs, not between a program and someone's reading of it.

That the legacy accepts explicit depot paths is what makes this possible.
The only reason it looked untestable is that production never passed them.
Nothing here touches the real depot: the paths handed over are temporary
files, and two tests below check the legacy actually read them.

It runs in a subprocess whose working directory is the production checkout,
which is how the scheduled task runs it. Its answer comes back as JSON and
is compared against ``evaluate_gate`` driven on the same payloads with the
extracted preset. The provenance of what was executed — ref, file digest,
resolved module path — is asserted alongside the result, so a passing run
states which legacy it agreed with.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lazystats.anomaly_gate import evaluate_gate
from lazystats.anomaly_gate_config import load_gate_config
from lazystats.io.depot import ResultDepot

#: The production checkout: the working directory the scheduled task uses,
#: and where the legacy gate still lives.
LAZYSTATS = Path(r"C:\Users\Administrator\Documents\GitHub\LazyStats")
LEGACY_GATE = LAZYSTATS / "anomaly_gate.py"

#: The private preset, whose values are the legacy module-level constants.
#: Equivalence only means something against a configuration that reproduces
#: them; what pins that is ``test_anomaly_preset_equivalence.py`` in the
#: private repository, not this file.
PRESET = Path(
    r"C:\Users\Administrator\Documents\GitHub\investmentcommittee"
    r"\config\daily_anomaly_gate.toml"
)

#: The identities the legacy selects rows by. Not choices this test makes:
#: they are the legacy's own constants, and the fixtures must be written
#: under them or the legacy finds nothing and "agrees" vacuously.
PRODUCED_BY = "scheduled:etf_daily_stats"
EXPLAINER_PRODUCED_BY = "lazystats.anomaly_explainer"

#: Driver for the subprocess. Run with ``-c`` and cwd set to the production
#: checkout, so ``import anomaly_gate`` resolves the same file the scheduled
#: task imports — without this test putting anything on sys.path.
DRIVER = """
import dataclasses, json, sys
import anomaly_gate
targets = anomaly_gate.find_investigation_targets(
    depot_path=sys.argv[1], explanations_depot_path=sys.argv[2]
)
print("MODULE:" + anomaly_gate.__file__)
print("RESULT:" + json.dumps([dataclasses.asdict(t) for t in targets], sort_keys=True))
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(*, as_of="2026-08-10", outliers=(), vol_short=None, vol_long=None,
            corr=None, returns=None) -> dict:
    """One ``etf_daily_stats`` payload, in the shape the ETF job writes."""
    return {
        "as_of": as_of,
        "outliers_last5": {"outliers": list(outliers)},
        "volatility_short": {"volatility": vol_short or {}},
        "volatility_long": {"volatility": vol_long or {}},
        "correlation_short": {"correlation": corr or {}},
        "returns_table": returns or {},
    }


def vol(ann=None, period=None) -> dict:
    out = {}
    if ann is not None:
        out["annualized_volatility"] = ann
    if period is not None:
        out["period_volatility"] = period
    return out


def outlier(instrument, date, z, ret, direction="down") -> dict:
    return {"instrument": instrument, "date": date, "z_score": z,
            "log_return": ret, "direction": direction}


def build_depots(tmp_path: Path, *, current: dict, previous: dict,
                 investigated: list[dict] | None = None) -> tuple[Path, Path, str]:
    """Two real depots holding the fixtures, written through the real API.

    ``previous`` is saved first: the legacy takes the two most recent rows by
    ``created_at`` and treats the newer as today's, so insertion order is
    part of the fixture rather than an implementation detail to gloss over.

    Returns the two paths and the result_id of the newer row, which is the
    trigger id the legacy will attach to its targets — read back from the
    depot rather than guessed, since it is generated on save.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    stats = tmp_path / "stats_depot.sqlite"
    explanations = tmp_path / "explanations_depot.sqlite"

    depot = ResultDepot(str(stats))
    try:
        for body in (previous, current):
            depot.save(
                kind="report", produced_by=PRODUCED_BY,
                instruments=sorted(
                    {o["instrument"] for o in body["outliers_last5"]["outliers"]}
                ),
                payload=body, provenance={"fixture": True},
                cadence="stable", series_key="etf_daily_stats",
            )
        newest = depot.list(produced_by=PRODUCED_BY, cadence="stable", limit=1)[0]
    finally:
        depot.close()

    depot = ResultDepot(str(explanations))
    try:
        if investigated:
            depot.save(
                kind="report", produced_by=EXPLAINER_PRODUCED_BY,
                instruments=[], payload={"items": investigated},
                provenance={"fixture": True}, cadence="stable",
                series_key="anomaly_explanations",
            )
    finally:
        depot.close()

    return stats, explanations, newest["result_id"]


def run_legacy(stats: Path, explanations: Path) -> tuple[list[dict], str]:
    """Execute the unmodified legacy gate; return what it selected and from where."""
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, str(stats), str(explanations)],
        cwd=str(LAZYSTATS), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"the legacy gate failed:\n{proc.stderr}"
    fields = {}
    for line in proc.stdout.splitlines():
        for tag in ("MODULE:", "RESULT:"):
            if line.startswith(tag):
                fields[tag[:-1]] = line[len(tag):]
    module = fields["MODULE"]
    assert Path(module).resolve() == LEGACY_GATE.resolve(), (
        f"the subprocess imported {module}, not the production gate"
    )
    return json.loads(fields["RESULT"]), module


def canonical(targets) -> list[dict]:
    """Both sides reduced to one comparable shape.

    Not a re-implementation of either: the legacy's dataclasses arrive via
    ``dataclasses.asdict`` and the new gate's via its own ``as_dict``. All
    this does is normalise the items sequence, which one expresses as a list
    and the other as a tuple.
    """
    out = []
    for t in targets:
        d = t if isinstance(t, dict) else t.as_dict()
        out.append({
            "date": d["date"],
            "trigger_result_id": d["trigger_result_id"],
            "items": [
                {"instrument": i["instrument"], "anomaly_type": i["anomaly_type"],
                 "date": i["date"], "detail": i["detail"]}
                for i in d["items"]
            ],
        })
    return out


@pytest.fixture(scope="module")
def cfg():
    return load_gate_config(PRESET)


@pytest.fixture(scope="module")
def provenance() -> dict:
    """What was executed, recorded rather than assumed."""
    ref = subprocess.run(
        ["git", "-C", str(LAZYSTATS), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {"ref": ref, "file": str(LEGACY_GATE), "sha256": digest(LEGACY_GATE),
            "cwd": str(LAZYSTATS)}


#: Every fixture the two implementations are compared on. Each trips a
#: different branch of the selection; the names are what a failure reports.
CASES: dict[str, dict] = {
    "quiet_day": {
        "current": payload(),
        "previous": payload(as_of="2026-08-07"),
    },
    "return_outlier": {
        "current": payload(outliers=[outlier("ticker:SPY", "2026-08-10", 3.1, -0.041)]),
        "previous": payload(as_of="2026-08-07"),
    },
    "outlier_on_a_weekend_is_dropped": {
        "current": payload(outliers=[outlier("ticker:SPY", "2026-08-08", 4.0, -0.05)]),
        "previous": payload(as_of="2026-08-07"),
    },
    "outlier_already_investigated": {
        "current": payload(outliers=[outlier("ticker:SPY", "2026-08-10", 3.1, -0.041)]),
        "previous": payload(as_of="2026-08-07"),
        # Stored without the "ticker:" prefix, which is how the explainer
        # writes them back — the mismatch that once made dedup never fire.
        "investigated": [{"anomaly_type": "return_outlier", "instrument": "SPY",
                          "date": "2026-08-10"}],
    },
    "several_outliers_across_two_dates": {
        "current": payload(outliers=[
            outlier("ticker:SPY", "2026-08-10", 3.1, -0.041),
            outlier("ticker:TLT", "2026-08-07", 2.6, 0.03, direction="up"),
            outlier("ticker:GLD", "2026-08-10", 2.2, -0.02),
        ]),
        "previous": payload(as_of="2026-08-07"),
    },
    "volatility_above_the_band_with_a_fresh_move": {
        "current": payload(vol_short={"ticker:QQQ": vol(ann=0.30)},
                           vol_long={"ticker:QQQ": vol(ann=0.15)}),
        "previous": payload(as_of="2026-08-07",
                            vol_short={"ticker:QQQ": vol(ann=0.16)},
                            vol_long={"ticker:QQQ": vol(ann=0.15)}),
    },
    "volatility_parked_in_the_band_without_a_fresh_move": {
        "current": payload(vol_short={"ticker:QQQ": vol(ann=0.30)},
                           vol_long={"ticker:QQQ": vol(ann=0.15)}),
        "previous": payload(as_of="2026-08-07",
                            vol_short={"ticker:QQQ": vol(ann=0.299)},
                            vol_long={"ticker:QQQ": vol(ann=0.15)}),
    },
    "volatility_below_the_band": {
        "current": payload(vol_short={"ticker:IEF": vol(ann=0.03)},
                           vol_long={"ticker:IEF": vol(ann=0.12)}),
        "previous": payload(as_of="2026-08-07",
                            vol_short={"ticker:IEF": vol(ann=0.11)},
                            vol_long={"ticker:IEF": vol(ann=0.12)}),
    },
    "correlation_break": {
        "current": payload(corr={"ticker:SPY": {"ticker:TLT": -0.55}}),
        "previous": payload(as_of="2026-08-07",
                            corr={"ticker:SPY": {"ticker:TLT": 0.10}}),
    },
    "correlation_pair_reported_once": {
        "current": payload(corr={"ticker:SPY": {"ticker:TLT": 0.95},
                                 "ticker:TLT": {"ticker:SPY": 0.95}}),
        "previous": payload(as_of="2026-08-07",
                            corr={"ticker:SPY": {"ticker:TLT": 0.30},
                                  "ticker:TLT": {"ticker:SPY": 0.30}}),
    },
    "correlation_cap": {
        "current": payload(corr={
            f"ticker:A{i}": {f"ticker:B{i}": 0.99} for i in range(12)
        }),
        "previous": payload(as_of="2026-08-07", corr={
            f"ticker:A{i}": {f"ticker:B{i}": 0.05} for i in range(12)
        }),
    },
    "beta_divergence": {
        # beta = 0.8 * (0.05 / 0.02) = 2.0; expected = 0.02; residual = -0.08;
        # residual vol = 0.05 * sqrt(1 - 0.64) = 0.03; z = -2.67, past the
        # threshold of 2.0, and the prior day's z is 0.
        "current": payload(
            vol_short={"ticker:SPY": vol(period=0.02), "ticker:ARKK": vol(period=0.05)},
            corr={"ticker:ARKK": {"ticker:SPY": 0.8}},
            returns={"ticker:SPY": {"1W": {"return": 0.01}},
                     "ticker:ARKK": {"1W": {"return": -0.06}}},
        ),
        "previous": payload(
            as_of="2026-08-07",
            vol_short={"ticker:SPY": vol(period=0.02), "ticker:ARKK": vol(period=0.05)},
            corr={"ticker:ARKK": {"ticker:SPY": 0.8}},
            returns={"ticker:SPY": {"1W": {"return": 0.01}},
                     "ticker:ARKK": {"1W": {"return": 0.02}}},
        ),
    },
    "everything_at_once": {
        "current": payload(
            outliers=[outlier("ticker:SPY", "2026-08-10", 3.1, -0.041)],
            vol_short={"ticker:QQQ": vol(ann=0.30)}, vol_long={"ticker:QQQ": vol(ann=0.15)},
            corr={"ticker:SPY": {"ticker:TLT": -0.55}},
        ),
        "previous": payload(
            as_of="2026-08-07",
            vol_short={"ticker:QQQ": vol(ann=0.16)}, vol_long={"ticker:QQQ": vol(ann=0.15)},
            corr={"ticker:SPY": {"ticker:TLT": 0.10}},
        ),
    },
}


def already_from(case: dict) -> frozenset[tuple[str, str]]:
    """The dedup set the caller now supplies, in the form the legacy derives
    internally: stored items carry no ``ticker:`` prefix and the legacy
    re-adds it before matching."""
    return frozenset(
        (i["instrument"] if i["instrument"].startswith("ticker:")
         else f"ticker:{i['instrument']}", i["date"])
        for i in case.get("investigated", [])
        if i["anomaly_type"] == "return_outlier"
    )


class TestEquivalenceAgainstTheExecutedLegacy:

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_both_gates_select_the_same_targets(self, tmp_path, cfg, name):
        case = CASES[name]
        stats, explanations, trigger_id = build_depots(
            tmp_path, current=case["current"], previous=case["previous"],
            investigated=case.get("investigated"),
        )
        legacy_targets, _ = run_legacy(stats, explanations)
        new_targets = evaluate_gate(
            current=case["current"], previous=case["previous"],
            trigger_result_id=trigger_id, config=cfg,
            already_investigated=already_from(case),
        )
        assert canonical(new_targets) == canonical(legacy_targets)


class TestTheComparisonIsNotVacuous:
    """A comparison where both sides always return nothing would pass."""

    def test_the_loaded_case_really_produces_targets(self, tmp_path, cfg):
        case = CASES["everything_at_once"]
        stats, explanations, _ = build_depots(
            tmp_path, current=case["current"], previous=case["previous"])
        legacy_targets, _ = run_legacy(stats, explanations)
        assert legacy_targets, "the legacy selected nothing even on the loaded case"
        assert sum(len(t["items"]) for t in legacy_targets) >= 3

    def test_each_anomaly_type_is_produced_by_some_case(self, tmp_path, cfg):
        """Otherwise a whole branch could have been silently dropped in the
        refactor and every comparison would still match on empty."""
        produced = set()
        for name in ("return_outlier", "volatility_above_the_band_with_a_fresh_move",
                     "correlation_break", "beta_divergence"):
            case = CASES[name]
            stats, explanations, _ = build_depots(
                tmp_path / name, current=case["current"], previous=case["previous"])
            targets, _ = run_legacy(stats, explanations)
            produced.update(i["anomaly_type"] for t in targets for i in t["items"])
        assert produced == {"return_outlier", "volatility_shift",
                            "correlation_shift", "beta_divergence"}

    def test_the_legacy_read_the_temporary_depot_and_not_the_real_one(self, tmp_path, cfg):
        """The fixtures use instruments and dates chosen here. Had the legacy
        fallen back to the production depot it would report something else,
        or nothing at all."""
        case = CASES["return_outlier"]
        stats, explanations, _ = build_depots(
            tmp_path, current=case["current"], previous=case["previous"])
        legacy_targets, _ = run_legacy(stats, explanations)
        assert [t["date"] for t in legacy_targets] == ["2026-08-10"]
        assert legacy_targets[0]["items"][0]["instrument"] == "ticker:SPY"

    def test_a_different_configuration_makes_the_two_disagree(self, tmp_path, cfg):
        """If they agreed whatever the thresholds, the equivalence would be
        measuring nothing. Raising the required fresh move past the fixture's
        must break the match."""
        case = CASES["volatility_above_the_band_with_a_fresh_move"]
        stats, explanations, trigger_id = build_depots(
            tmp_path, current=case["current"], previous=case["previous"])
        legacy_targets, _ = run_legacy(stats, explanations)
        mismatched = evaluate_gate(
            current=case["current"], previous=case["previous"],
            trigger_result_id=trigger_id,
            config=replace(cfg, vol_ratio_delta_min=10.0),
        )
        assert legacy_targets
        assert canonical(mismatched) != canonical(legacy_targets)


class TestProvenanceOfWhatWasCompared:
    """A passing equivalence must say which legacy it agreed with."""

    def test_the_legacy_file_is_present_and_identified(self, provenance):
        assert LEGACY_GATE.is_file()
        assert len(provenance["sha256"]) == 64
        assert len(provenance["ref"]) == 40

    def test_the_recorded_digest_is_of_the_file_that_was_imported(self, tmp_path,
                                                                 cfg, provenance):
        case = CASES["quiet_day"]
        stats, explanations, _ = build_depots(
            tmp_path, current=case["current"], previous=case["previous"])
        _, module = run_legacy(stats, explanations)
        assert digest(Path(module)) == provenance["sha256"]


class TestPropertiesTheShadowRestsOn:

    def test_the_new_gate_reaches_no_database(self):
        """Read off the module's imports, not by patching: the shadow plan
        must not be able to open a store even by accident."""
        source = (Path(__file__).resolve().parents[1] / "src" / "lazystats"
                  / "anomaly_gate.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & {"sqlite3", "os", "lazytools", "requests", "lazybridge"})

    def test_repeated_evaluation_is_deterministic(self, cfg):
        case = CASES["everything_at_once"]
        first = evaluate_gate(current=case["current"], previous=case["previous"],
                              trigger_result_id="res_x", config=cfg)
        second = evaluate_gate(current=case["current"], previous=case["previous"],
                               trigger_result_id="res_x", config=cfg)
        assert canonical(first) == canonical(second)
