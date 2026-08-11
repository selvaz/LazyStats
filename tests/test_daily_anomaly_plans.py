# -*- coding: utf-8 -*-
"""The two plans, the shadow runner's guards, and the explanation contract.

The question is not whether the flags parse but whether a shadow run can
reach a model, a database or a messenger. That is answered by inspecting the
plan the runner builds and, in a fresh process, which modules loading it
actually pulls in — never by patching something at runtime, since a
monkeypatched model would only prove the patch worked.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lazystats.anomaly_explanation import ExplanationError, validate_explanations
from lazystats.anomaly_gate_config import load_gate_config

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_daily_anomaly.py"
EXAMPLE = ROOT / "examples" / "daily_anomaly_gate.example.toml"


def load_runner():
    """Load the root-level runner as a module.

    Registered in sys.modules before execution: the runner defines a
    dataclass, and @dataclass resolves its owning module through
    sys.modules, failing on one that is not there.
    """
    name = "_daily_anomaly_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def runner():
    return load_runner()


@pytest.fixture(scope="module")
def cfg():
    return load_gate_config(EXAMPLE)


def payload(as_of="2026-08-10", outliers=()):
    return {
        "as_of": as_of,
        "outliers_last5": {"outliers": list(outliers)},
        "volatility_short": {"volatility": {}},
        "volatility_long": {"volatility": {}},
        "correlation_short": {"correlation": {}},
        "returns_table": {},
    }


@pytest.fixture
def input_artifact(tmp_path):
    p = tmp_path / "input.json"
    p.write_text(json.dumps({
        "trigger_result_id": "res_x",
        "current": payload(outliers=[{"instrument": "SPY", "date": "2026-08-10",
                                      "z_score": 3.0, "log_return": -0.04,
                                      "direction": "down"}]),
        "previous": payload(),
        "already_investigated": [],
    }), encoding="utf-8")
    return p


def context(runner, cfg, input_artifact, out_dir, mode="gate-shadow"):
    return runner.RunContext(
        config=cfg, upstream_series_key="example_daily_stats",
        upstream_produced_by=cfg.upstream_produced_by,
        input_artifact=input_artifact, output_dir=Path(out_dir), mode=mode,
    )


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(RUNNER), *args],
                          capture_output=True, text=True, cwd=str(ROOT), env=env)


class TestShadowPlanCannotReachProduction:
    def test_the_shadow_plan_contains_no_live_step(self, runner, cfg, input_artifact, tmp_path):
        """Absent, not disabled: a step that is not in the plan cannot run."""
        plan = runner.build_shadow_plan(context(runner, cfg, input_artifact, tmp_path / "o"))
        names = {name for name, _ in plan}
        assert not (names & runner.LIVE_ONLY_STEPS)
        assert names == {"gate", "write_gate_artifact"}

    def test_the_live_plan_does_contain_them(self, runner, cfg, input_artifact, tmp_path):
        """The complement: the guard above means something only if the live
        plan is really the one carrying those steps."""
        noop = lambda *a, **k: None  # noqa: E731
        plan = runner.build_live_plan(context(runner, cfg, input_artifact, tmp_path / "o"),
                                      explain=noop, persist=noop, render=noop, send=noop)
        names = {name for name, _ in plan}
        assert runner.LIVE_ONLY_STEPS <= names

    def test_both_plans_start_with_the_same_gate(self, runner, cfg, input_artifact, tmp_path):
        """Otherwise a shadow comparison would measure two different methods."""
        noop = lambda *a, **k: None  # noqa: E731
        shadow = [n for n, _ in runner.build_shadow_plan(
            context(runner, cfg, input_artifact, tmp_path / "a"))]
        live = [n for n, _ in runner.build_live_plan(
            context(runner, cfg, input_artifact, tmp_path / "b"),
            explain=noop, persist=noop, render=noop, send=noop)]
        assert shadow[0] == live[0] == "gate"

    def test_loading_the_runner_pulls_in_no_model_database_or_messenger(self, tmp_path):
        """Checked in a fresh process: an in-process check would be
        meaningless once the session has imported these for other reasons."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            f'''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("r", r"{RUNNER}")
m = importlib.util.module_from_spec(spec)
sys.modules["r"] = m
spec.loader.exec_module(m)
from lazystats.anomaly_gate_config import load_gate_config
cfg = load_gate_config(r"{EXAMPLE}")
ctx = m.RunContext(config=cfg, upstream_series_key="k",
                   upstream_produced_by=cfg.upstream_produced_by,
                   input_artifact=__import__("pathlib").Path("x.json"),
                   output_dir=__import__("pathlib").Path(r"{tmp_path}"),
                   mode="gate-shadow")
m.build_shadow_plan(ctx)
bad = [n for n in sys.modules
       if "claude" in n.lower() or "telegram" in n.lower()
       or "io.depot" in n or n.startswith("lazybridge")
       or n.startswith("requests") or n.startswith("sqlite3")]
print("LOADED:" + ",".join(sorted(bad)))
''', encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([sys.executable, str(probe)], capture_output=True,
                           text=True, cwd=str(ROOT), env=env)
        assert r.returncode == 0, r.stderr
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("LOADED:"))
        loaded = [x for x in line[len("LOADED:"):].split(",") if x]
        assert loaded == [], f"the shadow path pulled in {loaded}"


class TestShadowOutput:
    def test_writes_exactly_one_artifact(self, runner, cfg, input_artifact, tmp_path):
        out = tmp_path / "out"
        for _n, step in runner.build_shadow_plan(context(runner, cfg, input_artifact, out)):
            result = step()
        produced = list(out.glob("*"))
        assert len(produced) == 1
        assert produced[0].name == "anomaly_gate_2026-08-10.json"
        assert result["anomaly_count"] == 1

    def test_the_artifact_records_the_parameters_that_produced_it(self, runner, cfg,
                                                                  input_artifact, tmp_path):
        out = tmp_path / "out"
        for _n, step in runner.build_shadow_plan(context(runner, cfg, input_artifact, out)):
            step()
        art = json.loads((out / "anomaly_gate_2026-08-10.json").read_text(encoding="utf-8"))
        assert art["gate_parameters"]["vol_ratio_high"] == cfg.vol_ratio_high
        assert art["upstream_series_key"] == "example_daily_stats"
        assert art["trigger_result_id"] == "res_x"

    def test_two_runs_produce_identical_bytes(self, runner, cfg, input_artifact, tmp_path):
        """A wall-clock field would break comparison for reasons unrelated
        to the gate."""
        texts = []
        for name in ("a", "b"):
            out = tmp_path / name
            for _n, step in runner.build_shadow_plan(context(runner, cfg, input_artifact, out)):
                step()
            texts.append((out / "anomaly_gate_2026-08-10.json").read_text(encoding="utf-8"))
        assert texts[0] == texts[1]

    def test_nothing_is_written_outside_the_output_directory(self, runner, cfg,
                                                             input_artifact, tmp_path):
        out = tmp_path / "out"
        before = {p for p in ROOT.rglob("*") if p.is_file()}
        for _n, step in runner.build_shadow_plan(context(runner, cfg, input_artifact, out)):
            step()
        after = {p for p in ROOT.rglob("*") if p.is_file()}
        assert before == after, "the shadow run wrote inside the repository"


class TestCliGuards:
    def _base(self, tmp_path, input_artifact, out="out"):
        return ["--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                "--output-dir", str(tmp_path / out),
                "--upstream-series-key", "example_daily_stats"]

    def test_config_is_required(self, tmp_path, input_artifact):
        r = run_cli("--input-artifact", str(input_artifact),
                    "--output-dir", str(tmp_path / "o"),
                    "--upstream-series-key", "k")
        assert r.returncode == 2 and "--config" in (r.stderr + r.stdout)

    def test_a_missing_config_names_it(self, tmp_path, input_artifact):
        r = run_cli("--config", "absent.toml", "--input-artifact", str(input_artifact),
                    "--output-dir", str(tmp_path / "o"), "--upstream-series-key", "k")
        assert r.returncode == 2 and "not found" in r.stderr

    def test_a_missing_input_artifact_fails(self, tmp_path):
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(tmp_path / "nope.json"),
                    "--output-dir", str(tmp_path / "o"), "--upstream-series-key", "k")
        assert r.returncode == 1 and "input artifact not found" in r.stderr

    def test_a_corrupt_input_artifact_fails(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(bad),
                    "--output-dir", str(tmp_path / "o"), "--upstream-series-key", "k")
        assert r.returncode == 1 and "not valid JSON" in r.stderr

    def test_an_input_missing_a_required_key_fails(self, tmp_path):
        bad = tmp_path / "partial.json"
        bad.write_text(json.dumps({"current": payload()}), encoding="utf-8")
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(bad),
                    "--output-dir", str(tmp_path / "o"), "--upstream-series-key", "k")
        assert r.returncode == 1 and "missing" in r.stderr

    def test_a_non_empty_output_dir_is_refused(self, tmp_path, input_artifact):
        used = tmp_path / "used"
        used.mkdir()
        (used / "stale.json").write_text("{}", encoding="utf-8")
        r = run_cli(*self._base(tmp_path, input_artifact, out="used"))
        assert r.returncode == 2 and "new or empty" in r.stderr

    def test_the_protected_directory_is_refused(self, tmp_path, input_artifact):
        protected = tmp_path / "reports"
        protected.mkdir()
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(protected), "--upstream-series-key", "k",
                    "--protected-dir", str(protected))
        assert r.returncode == 2 and "must not be inside" in r.stderr

    def test_a_subdirectory_of_the_protected_directory_is_refused(self, tmp_path, input_artifact):
        """reports/shadow/ is still production output."""
        protected = tmp_path / "reports"
        protected.mkdir()
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(protected / "shadow"), "--upstream-series-key", "k",
                    "--protected-dir", str(protected))
        assert r.returncode == 2 and "must not be inside" in r.stderr

    def test_a_valid_run_succeeds(self, tmp_path, input_artifact):
        r = run_cli(*self._base(tmp_path, input_artifact))
        assert r.returncode == 0, r.stderr
        assert "anomaly_count" in r.stdout


class TestLivePlanUsesAnInjectedEngine:
    def test_the_live_plan_calls_what_it_is_given(self, runner, cfg, input_artifact, tmp_path):
        """No real model: a deterministic fake, passed explicitly."""
        calls = []

        def fake_explain(artifact):
            calls.append("explain")
            anomalies = [i for t in artifact["targets"] for i in t["items"]]
            return {"artifact": artifact, "explanations": [
                {"instrument": a["instrument"], "date": a["date"],
                 "anomaly_type": a["anomaly_type"], "category": "macro_data",
                 "explanation": "a deterministic fake", "confidence": 0.5}
                for a in anomalies]}

        for label in ("persist", "render", "send"):
            pass
        plan = runner.build_live_plan(
            context(runner, cfg, input_artifact, tmp_path / "o"),
            explain=fake_explain,
            persist=lambda x: calls.append("persist"),
            render=lambda x: calls.append("render"),
            send=lambda x: calls.append("send"),
        )
        for _n, step in plan:
            step()
        assert calls == ["explain", "persist", "render", "send"]


class TestExplanationContract:
    def _anomalies(self):
        return [{"instrument": "SPY", "date": "2026-08-10", "anomaly_type": "return_outlier"},
                {"instrument": "TLT", "date": "2026-08-10", "anomaly_type": "volatility_shift"}]

    def _valid(self):
        return [{"instrument": "SPY", "date": "2026-08-10", "anomaly_type": "return_outlier",
                 "category": "macro_data", "explanation": "CPI surprise", "confidence": 0.7},
                {"instrument": "TLT", "date": "2026-08-10", "anomaly_type": "volatility_shift",
                 "category": "monetary_policy", "explanation": "rate repricing",
                 "confidence": 0.4, "evidence": ["https://example.invalid/a"]}]

    def test_a_well_formed_response_is_accepted(self):
        out = validate_explanations(self._valid(), anomalies=self._anomalies())
        assert len(out) == 2
        assert out[0].instrument == "SPY", "returned in the order the anomalies were given"

    def test_an_unknown_category_is_refused(self):
        bad = self._valid()
        bad[0]["category"] = "vibes"
        with pytest.raises(ExplanationError, match="not one of"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_an_explanation_for_an_anomaly_never_sent_is_refused(self):
        """A model will happily explain something nobody asked about."""
        bad = self._valid()
        bad[0]["instrument"] = "GLD"
        with pytest.raises(ExplanationError, match="not among the anomalies sent"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_missing_explanation_is_refused_not_trimmed(self):
        """Reporting the rest would hide that the model skipped one."""
        with pytest.raises(ExplanationError, match="no explanation for"):
            validate_explanations(self._valid()[:1], anomalies=self._anomalies())

    def test_a_duplicated_explanation_is_refused(self):
        bad = self._valid() + [self._valid()[0]]
        with pytest.raises(ExplanationError, match="more than once"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_an_unknown_field_is_refused(self):
        bad = self._valid()
        bad[0]["recommendation"] = "sell everything"
        with pytest.raises(ExplanationError, match="unknown field"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_missing_required_field_is_refused(self):
        bad = self._valid()
        del bad[0]["confidence"]
        with pytest.raises(ExplanationError, match="missing required field"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_blank_explanation_is_refused(self):
        bad = self._valid()
        bad[0]["explanation"] = "   "
        with pytest.raises(ExplanationError, match="must not be blank"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_confidence_outside_zero_to_one_is_refused(self):
        bad = self._valid()
        bad[0]["confidence"] = 1.5
        with pytest.raises(ExplanationError, match=r"\[0, 1\]"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_boolean_confidence_is_refused(self):
        """bool passes isinstance(int); `true` is not a confidence of 1.0."""
        bad = self._valid()
        bad[0]["confidence"] = True
        with pytest.raises(ExplanationError, match="confidence"):
            validate_explanations(bad, anomalies=self._anomalies())

    def test_a_non_list_response_is_refused(self):
        with pytest.raises(ExplanationError, match="expected a list"):
            validate_explanations({"explanations": []}, anomalies=self._anomalies())

    def test_evidence_must_be_a_list_of_strings(self):
        bad = self._valid()
        bad[0]["evidence"] = [{"url": "x"}]
        with pytest.raises(ExplanationError, match="list of strings"):
            validate_explanations(bad, anomalies=self._anomalies())
