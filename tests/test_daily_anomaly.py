"""The two plans, the shadow guards, and the explanation contract.

Self-contained. Every path is a ``tmp_path``, every threshold comes from the
example preset shipped in this repository, and every instrument is a made-up
name. Nothing here reaches outside the checkout, so the suite runs the same
on any machine.

The question these tests answer is not whether the flags parse but whether a
shadow run can reach a model, a database or a messenger, and whether it can
write anywhere it was told not to. That is settled by inspecting the plan the
runner builds and, in a fresh process, which modules importing it actually
pulls in — never by patching something at runtime, since a monkeypatched
model would only prove the patch worked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lazystats.anomaly_explanation import (
    ExplanationError,
    validate_batch,
    validate_explanations,
)
from lazystats.anomaly_gate_config import load_gate_config
from lazystats.daily_anomaly import (
    LIVE_ONLY_STEPS,
    RunContext,
    RunError,
    build_live_plan,
    build_shadow_plan,
    is_inside,
    load_input_artifact,
    validate_as_of,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_daily_anomaly.py"
EXAMPLE = ROOT / "examples" / "daily_anomaly_gate.example.toml"


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


def write_input(path: Path, *, as_of="2026-08-10", outliers=None, **extra) -> Path:
    body = {
        "trigger_result_id": "res_example",
        "current": payload(as_of, outliers if outliers is not None else [
            {"instrument": "ticker:AAA", "date": "2026-08-10", "z_score": 4.0,
             "log_return": -0.06, "direction": "down"},
        ]),
        "previous": payload("2026-08-07"),
        "already_investigated": [],
    }
    body.update(extra)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def input_artifact(tmp_path):
    return write_input(tmp_path / "input.json")


def context(cfg, input_artifact, out_dir, protected=None):
    return RunContext(
        config=cfg,
        input_artifact=input_artifact,
        output_dir=Path(out_dir),
        protected_dirs=tuple(Path(p) for p in (protected or [out_dir.parent / "reports"])),
    )


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(RUNNER), *args],
                          capture_output=True, text=True, cwd=str(ROOT), env=env)


class TestShadowPlanCannotReachProduction:
    def test_the_shadow_plan_contains_no_live_step(self, cfg, input_artifact, tmp_path):
        """Absent, not disabled: a step that is not in the plan cannot run."""
        plan = build_shadow_plan(context(cfg, input_artifact, tmp_path / "o"))
        names = {name for name, _ in plan}
        assert not (names & LIVE_ONLY_STEPS)
        assert names == {"gate", "write_gate_artifact"}

    def test_the_live_plan_does_contain_them(self, cfg, input_artifact, tmp_path):
        """The complement: the guard above means something only if the live
        plan is really the one carrying those steps."""
        noop = lambda *a, **k: None  # noqa: E731
        plan = build_live_plan(context(cfg, input_artifact, tmp_path / "o"),
                               explain=noop, persist=noop, render=noop, send=noop)
        assert LIVE_ONLY_STEPS <= {name for name, _ in plan}

    def test_both_plans_start_with_the_same_gate(self, cfg, input_artifact, tmp_path):
        """Otherwise a shadow comparison would measure two different methods."""
        noop = lambda *a, **k: None  # noqa: E731
        shadow = [n for n, _ in build_shadow_plan(context(cfg, input_artifact, tmp_path / "a"))]
        live = [n for n, _ in build_live_plan(context(cfg, input_artifact, tmp_path / "b"),
                                              explain=noop, persist=noop, render=noop, send=noop)]
        assert shadow[0] == live[0] == "gate"

    def test_importing_the_module_pulls_in_no_model_database_or_messenger(self, tmp_path):
        """Checked in a fresh process: an in-process check would be
        meaningless once the session has imported these for other reasons.

        The module is imported normally — nothing is registered in
        ``sys.modules`` by hand, which would be the test arranging the very
        thing it claims to observe."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            'import sys\n'
            'import lazystats.daily_anomaly as m\n'
            'from lazystats.anomaly_gate_config import load_gate_config\n'
            f'cfg = load_gate_config(r"{EXAMPLE}")\n'
            'import pathlib\n'
            'ctx = m.RunContext(config=cfg, input_artifact=pathlib.Path("x.json"),\n'
            f'                  output_dir=pathlib.Path(r"{tmp_path}"),\n'
            f'                  protected_dirs=(pathlib.Path(r"{tmp_path / "p"}"),))\n'
            'm.build_shadow_plan(ctx)\n'
            'bad = [n for n in sys.modules\n'
            '       if "claude" in n.lower() or "telegram" in n.lower()\n'
            '       or "depot" in n or n.startswith("lazybridge")\n'
            '       or n.startswith("requests") or n.startswith("sqlite3")]\n'
            'print("LOADED:" + ",".join(sorted(bad)))\n',
            encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([sys.executable, str(probe)], capture_output=True,
                           text=True, cwd=str(ROOT), env=env)
        assert r.returncode == 0, r.stderr
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("LOADED:"))
        loaded = [x for x in line[len("LOADED:"):].split(",") if x]
        assert loaded == [], f"the shadow path pulled in {loaded}"


class TestShadowOutput:
    def test_writes_exactly_one_artifact(self, cfg, input_artifact, tmp_path):
        out = tmp_path / "out"
        result = None
        for _n, step in build_shadow_plan(context(cfg, input_artifact, out)):
            result = step()
        produced = list(out.glob("*"))
        assert len(produced) == 1
        assert produced[0].name == "anomaly_gate_2026-08-10.json"
        assert result["anomaly_count"] == 1

    def test_the_artifact_records_the_parameters_that_produced_it(self, cfg,
                                                                  input_artifact, tmp_path):
        out = tmp_path / "out"
        for _n, step in build_shadow_plan(context(cfg, input_artifact, out)):
            step()
        art = json.loads((out / "anomaly_gate_2026-08-10.json").read_text(encoding="utf-8"))
        assert art["gate_parameters"]["vol_ratio_high"] == cfg.vol_ratio_high
        assert art["upstream_series_key"] == cfg.upstream_series_key
        assert art["upstream_produced_by"] == cfg.upstream_produced_by
        assert art["trigger_result_id"] == "res_example"

    def test_the_series_identity_comes_from_the_preset_alone(self, cfg, input_artifact,
                                                             tmp_path):
        """There is no command-line override, so an artifact cannot claim one
        series while the preset names another."""
        out = tmp_path / "out"
        for _n, step in build_shadow_plan(context(cfg, input_artifact, out)):
            step()
        art = json.loads((out / "anomaly_gate_2026-08-10.json").read_text(encoding="utf-8"))
        assert art["upstream_series_key"] == "example_daily_stats"
        assert "--upstream-series-key" not in RUNNER.read_text(encoding="utf-8")

    def test_two_runs_produce_identical_bytes(self, cfg, input_artifact, tmp_path):
        """A shadow artifact is compared against another; a wall-clock field
        would differ every run for reasons unrelated to the gate."""
        digests = []
        for name in ("a", "b"):
            out = tmp_path / name
            for _n, step in build_shadow_plan(context(cfg, input_artifact, out)):
                step()
            digests.append((out / "anomaly_gate_2026-08-10.json").read_bytes())
        assert digests[0] == digests[1]

    def test_nothing_is_written_outside_the_output_directory(self, cfg, input_artifact,
                                                             tmp_path):
        """Every file under the run's own root, before and after, with the
        authorised output excluded — so the check does not simply notice the
        artifact it asked for."""
        out = tmp_path / "out"
        before = {p for p in tmp_path.rglob("*") if p.is_file()}
        for _n, step in build_shadow_plan(context(cfg, input_artifact, out)):
            step()
        after = {p for p in tmp_path.rglob("*") if p.is_file()}
        created = {p for p in after - before if not is_inside(p, out)}
        assert created == set()


class TestFailClosed:
    """Refusals. Each one is a way a shadow run could stop being harmless."""

    def test_an_output_directory_inside_a_protected_tree_is_refused(self, cfg,
                                                                    input_artifact, tmp_path):
        protected = tmp_path / "reports"
        protected.mkdir()
        ctx = context(cfg, input_artifact, protected / "sub", protected=[protected])
        with pytest.raises(RunError, match="protected"):
            for _n, step in build_shadow_plan(ctx):
                step()

    def test_the_protected_tree_itself_is_refused(self, cfg, input_artifact, tmp_path):
        protected = tmp_path / "reports"
        protected.mkdir()
        ctx = context(cfg, input_artifact, protected, protected=[protected])
        with pytest.raises(RunError, match="protected"):
            for _n, step in build_shadow_plan(ctx):
                step()

    def test_every_declared_protected_tree_is_checked_not_just_the_first(self, cfg,
                                                                        input_artifact,
                                                                        tmp_path):
        first = tmp_path / "one"
        second = tmp_path / "two"
        first.mkdir()
        second.mkdir()
        ctx = context(cfg, input_artifact, second / "sub", protected=[first, second])
        with pytest.raises(RunError, match="protected"):
            for _n, step in build_shadow_plan(ctx):
                step()

    def test_declaring_none_is_refused(self, cfg, input_artifact, tmp_path):
        """An empty tuple would make the guard vacuous, so it is not a valid
        way to call this."""
        ctx = RunContext(config=cfg, input_artifact=input_artifact,
                         output_dir=tmp_path / "out", protected_dirs=())
        with pytest.raises(RunError, match="at least one protected"):
            for _n, step in build_shadow_plan(ctx):
                step()

    @pytest.mark.parametrize("bad", [
        "../escape", "2026-08-10/../..", "..\\escape", "C:\\Windows\\evil",
        "2026-08-10\x00", "", "not-a-date", "2026-8-10", "20260810",
    ])
    def test_a_date_that_could_steer_a_path_is_refused(self, bad):
        with pytest.raises(RunError, match="as_of"):
            validate_as_of(bad)

    def test_a_traversing_date_writes_nothing(self, cfg, tmp_path):
        """Not only refused: the whole run must leave the filesystem alone."""
        artifact = write_input(tmp_path / "input.json", as_of="../../escape")
        out = tmp_path / "out"
        before = {p for p in tmp_path.rglob("*") if p.is_file()}
        with pytest.raises(RunError, match="as_of"):
            for _n, step in build_shadow_plan(context(cfg, artifact, out)):
                step()
        assert {p for p in tmp_path.rglob("*") if p.is_file()} == before

    def test_a_date_valid_in_shape_but_absurd_still_stays_inside(self, cfg, tmp_path):
        """The regex admits 9999-99-99; what matters is that it cannot leave
        the output directory."""
        artifact = write_input(tmp_path / "input.json", as_of="9999-99-99")
        out = tmp_path / "out"
        for _n, step in build_shadow_plan(context(cfg, artifact, out)):
            step()
        assert [p.name for p in out.iterdir()] == ["anomaly_gate_9999-99-99.json"]


class TestInputIsValidated:
    """A malformed input must fail here, not deep inside the gate."""

    def test_a_missing_file_is_a_run_error(self, tmp_path):
        with pytest.raises(RunError, match="not found"):
            load_input_artifact(tmp_path / "absent.json")

    def test_a_directory_is_a_run_error(self, tmp_path):
        """Reading a directory raises OSError, not FileNotFoundError."""
        with pytest.raises(RunError):
            load_input_artifact(tmp_path)

    def test_invalid_json_is_a_run_error(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text("{ not json", encoding="utf-8")
        with pytest.raises(RunError, match="not valid JSON"):
            load_input_artifact(p)

    @pytest.mark.parametrize("body,match", [
        ('[]', "must be a JSON object"),
        ('{"previous": {}, "trigger_result_id": "r"}', "missing 'current'"),
        ('{"current": {}, "trigger_result_id": "r"}', "missing 'previous'"),
        ('{"current": [], "previous": {}, "trigger_result_id": "r"}', "must be an object"),
        ('{"current": {"as_of": "2026-08-10"}, "previous": {}}', "trigger_result_id"),
        ('{"current": {"as_of": "2026-08-10"}, "previous": {}, "trigger_result_id": "  "}',
         "trigger_result_id"),
        ('{"current": {}, "previous": {}, "trigger_result_id": "r"}', "missing 'as_of'"),
        ('{"current": {"as_of": 20260810}, "previous": {}, "trigger_result_id": "r"}',
         "as_of"),
        ('{"current": {"as_of": "2026-08-10"}, "previous": {}, "trigger_result_id": "r",'
         ' "already_investigated": {}}', "must be a list"),
        ('{"current": {"as_of": "2026-08-10"}, "previous": {}, "trigger_result_id": "r",'
         ' "already_investigated": [{"instrument": "x"}]}', "already_investigated"),
    ])
    def test_a_malformed_input_is_refused_with_a_message_naming_the_field(
            self, tmp_path, body, match):
        p = tmp_path / "x.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(RunError, match=match):
            load_input_artifact(p)


class TestCommandLine:
    def test_a_protected_directory_is_mandatory(self, tmp_path, input_artifact):
        """Not optional: a run with nothing declared would pass every guard by
        having nothing to check against."""
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(tmp_path / "out"))
        assert r.returncode == 2
        assert "--protected-dir" in r.stderr

    def test_a_missing_config_is_refused(self, tmp_path, input_artifact):
        r = run_cli("--config", str(tmp_path / "absent.toml"),
                    "--input-artifact", str(input_artifact),
                    "--output-dir", str(tmp_path / "out"),
                    "--protected-dir", str(tmp_path / "reports"))
        assert r.returncode == 2
        assert "CONFIG ERROR" in r.stderr

    def test_a_non_empty_output_directory_is_refused(self, tmp_path, input_artifact):
        out = tmp_path / "out"
        out.mkdir()
        (out / "leftover.txt").write_text("x", encoding="utf-8")
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(out),
                    "--protected-dir", str(tmp_path / "reports"))
        assert r.returncode == 2
        assert "new or empty" in r.stderr

    def test_an_output_inside_a_protected_tree_is_refused(self, tmp_path, input_artifact):
        protected = tmp_path / "reports"
        protected.mkdir()
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(protected / "sub"),
                    "--protected-dir", str(protected))
        assert r.returncode == 2
        assert "protected" in r.stderr

    def test_a_missing_input_is_a_run_error_not_a_config_error(self, tmp_path):
        """Different exit codes: one says the job was asked for wrongly, the
        other that it could not be carried out."""
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(tmp_path / "absent.json"),
                    "--output-dir", str(tmp_path / "out"),
                    "--protected-dir", str(tmp_path / "reports"))
        assert r.returncode == 1
        assert "RUN ERROR" in r.stderr

    def test_a_good_run_succeeds_and_prints_the_artifact_path(self, tmp_path, input_artifact):
        out = tmp_path / "out"
        r = run_cli("--config", str(EXAMPLE), "--input-artifact", str(input_artifact),
                    "--output-dir", str(out),
                    "--protected-dir", str(tmp_path / "reports"))
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["anomaly_count"] == 1
        assert (out / "anomaly_gate_2026-08-10.json").is_file()


class TestLivePlanComposition:
    def test_the_injected_stages_run_in_order_on_the_gate_result(self, cfg, input_artifact,
                                                                 tmp_path):
        """A fake engine, injected. No model is constructed anywhere."""
        seen: list[str] = []

        def explain(artifact):
            seen.append("explain")
            return {"trigger_result_id": artifact["trigger_result_id"], "explanations": []}

        def record(name):
            def stage(_payload):
                seen.append(name)
            return stage

        plan = build_live_plan(context(cfg, input_artifact, tmp_path / "o"),
                               explain=explain, persist=record("persist"),
                               render=record("render"), send=record("send"))
        for _name, step in plan:
            step()
        assert seen == ["explain", "persist", "render", "send"]


class TestExplanationContract:
    def anomalies(self):
        return [{"instrument": "ticker:AAA", "date": "2026-08-10",
                 "anomaly_type": "return_outlier", "detail": {}}]

    def good(self, **overrides):
        item = {"instrument": "ticker:AAA", "date": "2026-08-10",
                "anomaly_type": "return_outlier", "category": "macro_data",
                "explanation": "A scheduled data release.", "confidence": 0.7}
        item.update(overrides)
        return item

    def test_a_well_formed_answer_is_accepted(self):
        out = validate_explanations([self.good()], anomalies=self.anomalies())
        assert out[0].category == "macro_data"

    @pytest.mark.parametrize("override,match", [
        ({"category": "invented"}, "not one of"),
        ({"confidence": 1.4}, r"\[0, 1\]"),
        ({"confidence": True}, "must be"),
        ({"explanation": "   "}, "must not be blank"),
        ({"instrument": "ticker:ZZZ"}, "not among the anomalies sent"),
    ])
    def test_a_bad_field_is_refused(self, override, match):
        with pytest.raises(ExplanationError, match=match):
            validate_explanations([self.good(**override)], anomalies=self.anomalies())

    def test_an_unknown_field_is_refused(self):
        item = self.good()
        item["sentiment"] = "bullish"
        with pytest.raises(ExplanationError, match="unknown field"):
            validate_explanations([item], anomalies=self.anomalies())

    def test_a_missing_explanation_is_rejected_not_trimmed(self):
        anomalies = self.anomalies() + [
            {"instrument": "ticker:BBB", "date": "2026-08-10",
             "anomaly_type": "return_outlier", "detail": {}}]
        with pytest.raises(ExplanationError, match="no explanation for"):
            validate_explanations([self.good()], anomalies=anomalies)

    def test_the_same_anomaly_explained_twice_is_refused(self):
        with pytest.raises(ExplanationError, match="more than once"):
            validate_explanations([self.good(), self.good()], anomalies=self.anomalies())


class TestExplanationBatchIsBoundToItsRun:
    """The envelope, which is what stops yesterday's answers being attached
    to today's gate result."""

    def artifact(self, trigger="res_example"):
        return {
            "trigger_result_id": trigger,
            "targets": [{"date": "2026-08-10", "trigger_result_id": trigger,
                         "items": [{"instrument": "ticker:AAA", "date": "2026-08-10",
                                    "anomaly_type": "return_outlier", "detail": {}}]}],
        }

    def response(self, trigger="res_example"):
        return {"trigger_result_id": trigger, "explanations": [
            {"instrument": "ticker:AAA", "date": "2026-08-10",
             "anomaly_type": "return_outlier", "category": "macro_data",
             "explanation": "A scheduled data release.", "confidence": 0.7}]}

    def test_a_matching_batch_is_accepted(self):
        batch = validate_batch(self.response(), artifact=self.artifact())
        assert batch.trigger_result_id == "res_example"
        assert len(batch.explanations) == 1

    def test_a_different_trigger_id_is_refused(self):
        with pytest.raises(ExplanationError, match="different run"):
            validate_batch(self.response("res_yesterday"), artifact=self.artifact())

    def test_a_missing_trigger_id_is_refused(self):
        body = self.response()
        del body["trigger_result_id"]
        with pytest.raises(ExplanationError, match="trigger_result_id"):
            validate_batch(body, artifact=self.artifact())

    def test_a_blank_trigger_id_is_refused(self):
        with pytest.raises(ExplanationError, match="trigger_result_id"):
            validate_batch(self.response("   "), artifact=self.artifact())

    def test_a_bare_list_is_refused(self):
        """The list form carries no identity, so it cannot be bound."""
        with pytest.raises(ExplanationError, match="expected an object"):
            validate_batch(self.response()["explanations"], artifact=self.artifact())

    def test_missing_explanations_key_is_refused(self):
        with pytest.raises(ExplanationError, match="missing 'explanations'"):
            validate_batch({"trigger_result_id": "res_example"}, artifact=self.artifact())

    def test_the_per_item_contract_still_applies_inside_the_envelope(self):
        body = self.response()
        body["explanations"][0]["category"] = "invented"
        with pytest.raises(ExplanationError, match="not one of"):
            validate_batch(body, artifact=self.artifact())
