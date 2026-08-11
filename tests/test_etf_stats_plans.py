# -*- coding: utf-8 -*-
"""The live and shadow plans, and the CLI guards around them.

The question these tests answer is not "does the flag parse" but "can a
shadow run reach production?". They inspect the plan the runner actually
builds and the files it actually writes — never by replacing functions at
runtime, because a monkeypatched depot would prove only that the patch
worked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_daily_etf_stats.py"
EXAMPLE = ROOT / "examples" / "etf_daily_stats.example.toml"


def load_runner():
    """Import the root-level runner as a module.

    It is a script beside the package, not part of it, so pytest's
    ``pythonpath`` does not cover it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the runner as a real process, as a scheduled task would.

    PYTHONPATH points at this checkout's src/ because pytest's own
    ``pythonpath`` setting does not reach a subprocess — without it the
    child would import whichever lazystats happens to be pip-installed.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )


@pytest.fixture
def runner():
    return load_runner()


@pytest.fixture
def cfg(runner):
    from lazystats.etf_stats import load_config
    return load_config(EXAMPLE)


class TestShadowPlanCannotReachProduction:
    def test_shadow_plan_omits_every_live_only_step(self, runner, cfg, tmp_path):
        """Not disabled — absent. A step that is not in the plan cannot run."""
        plan = runner.build_shadow_plan(cfg, str(tmp_path))
        names = {s.name for s in plan.steps}
        assert not (names & runner._LIVE_ONLY_STEPS), (
            f"shadow plan contains live-only steps: {names & runner._LIVE_ONLY_STEPS}"
        )

    def test_shadow_plan_ends_by_writing_its_own_outputs(self, runner, cfg, tmp_path):
        plan = runner.build_shadow_plan(cfg, str(tmp_path))
        assert [s.name for s in plan.steps][-1] == "write_shadow_outputs"

    def test_live_plan_does_contain_them(self, runner, cfg):
        """The complement: the guard above is meaningful only if the live
        plan really is the one carrying those steps."""
        names = {s.name for s in runner.build_live_plan(cfg).steps}
        assert runner._LIVE_ONLY_STEPS <= names

    def test_both_plans_share_the_same_analysis_steps(self, runner, cfg, tmp_path):
        """Otherwise a shadow comparison would be measuring two methods."""
        analysis = [s.name for s in runner._make_analysis_steps(cfg)]
        live = [s.name for s in runner.build_live_plan(cfg).steps]
        shadow = [s.name for s in runner.build_shadow_plan(cfg, str(tmp_path)).steps]
        assert live[: len(analysis)] == analysis
        assert shadow[: len(analysis)] == analysis


class TestShadowWriterIsSelfContained:
    def _bundle(self, cfg) -> str:
        """A synthetic bundle: no market data, no database, no network."""
        instruments = list(cfg.instruments)
        return json.dumps({
            "as_of": "2026-08-08",
            "instruments": instruments,
            "instrument_meta": [{"symbol": s, "name": s} for s in instruments],
            "volatility_short": {}, "volatility_long": {}, "volatility_1y": {},
            "correlation_short": {}, "correlation_long": {},
            "outliers_last5": [], "outlier_daily_counts": [],
            "returns_table": [],
        })

    def test_writes_json_and_html_into_the_given_directory(self, runner, cfg, tmp_path):
        out = tmp_path / "shadow-out"
        write = runner._make_write_shadow_outputs(cfg, str(out))
        info = write(self._bundle(cfg))

        assert Path(info["json_path"]).parent == out
        assert Path(info["html_path"]).parent == out
        assert Path(info["json_path"]).is_file()
        assert Path(info["html_path"]).is_file()

    def test_payload_has_the_canonical_row_shape(self, runner, cfg, tmp_path):
        """Same shape the depot would return, so the two paths are comparable."""
        write = runner._make_write_shadow_outputs(cfg, str(tmp_path))
        info = write(self._bundle(cfg))
        row = json.loads(Path(info["json_path"]).read_text(encoding="utf-8"))
        assert set(row) == {
            "result_id", "kind", "produced_by", "instruments", "payload",
            "provenance", "created_at", "cadence", "series_key",
        }
        assert row["series_key"] == cfg.series_key
        assert row["provenance"]["outlier_threshold"] == cfg.outlier_threshold

    def test_created_at_is_fixed_so_two_runs_compare_byte_for_byte(self, runner, cfg, tmp_path):
        write_a = runner._make_write_shadow_outputs(cfg, str(tmp_path / "a"))
        write_b = runner._make_write_shadow_outputs(cfg, str(tmp_path / "b"))
        a = Path(write_a(self._bundle(cfg))["json_path"]).read_text(encoding="utf-8")
        b = Path(write_b(self._bundle(cfg))["json_path"]).read_text(encoding="utf-8")
        assert a == b, "a wall-clock field would differ for reasons unrelated to the analysis"

    def test_writes_nowhere_else(self, runner, cfg, tmp_path):
        """Nothing lands in the production reports directory."""
        prod = Path(runner.REPORTS_DIR)
        before = set(prod.glob("*")) if prod.exists() else set()
        runner._make_write_shadow_outputs(cfg, str(tmp_path))(self._bundle(cfg))
        after = set(prod.glob("*")) if prod.exists() else set()
        assert before == after


class TestCliGuards:
    def test_config_is_required(self):
        r = run_cli("--as-of", "2026-08-08")
        assert r.returncode == 2
        assert "--config" in (r.stderr + r.stdout)

    def test_missing_config_file_names_it(self):
        r = run_cli("--config", "absent.toml", "--as-of", "2026-08-08")
        assert r.returncode == 2
        assert "not found" in r.stderr

    def test_dry_run_requires_an_output_dir(self):
        """Otherwise a shadow run would default into the live reports directory."""
        r = run_cli("--config", str(EXAMPLE), "--as-of", "2026-08-08", "--dry-run")
        assert r.returncode == 2
        assert "--output-dir" in r.stderr

    def test_dry_run_refuses_the_production_reports_directory(self, runner):
        r = run_cli("--config", str(EXAMPLE), "--as-of", "2026-08-08",
                    "--dry-run", "--output-dir", runner.REPORTS_DIR)
        assert r.returncode == 2
        assert "must not be the production reports directory" in r.stderr

    def test_output_dir_without_dry_run_is_refused(self, tmp_path):
        """A live run writes where production expects; accepting the flag
        silently would suggest otherwise."""
        r = run_cli("--config", str(EXAMPLE), "--as-of", "2026-08-08",
                    "--output-dir", str(tmp_path))
        assert r.returncode == 2
        assert "--dry-run only" in r.stderr


class TestNoPrivateDependency:
    def test_the_public_package_does_not_import_investmentcommittee(self):
        """The preset lives in the private repo; the method must not know it."""
        import ast

        offenders = []
        for module in (ROOT / "src" / "lazystats").rglob("*.py"):
            tree = ast.parse(module.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] == "investmentcommittee":
                        offenders.append(module.name)
                elif isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] == "investmentcommittee" for a in node.names):
                        offenders.append(module.name)
        assert not offenders, f"public package imports the private repo: {offenders}"

    def test_the_runner_carries_no_hardcoded_preset(self):
        """The whole point of the extraction."""
        text = RUNNER.read_text(encoding="utf-8")
        for name in ("INSTRUMENTS = [", "OUTLIER_THRESHOLD =", "SHORT_WEEKS =",
                     "LONG_WEEKS =", "SERIES_KEY =", "RETURN_HORIZONS = ["):
            assert name not in text, f"{name.strip(' =[')} is back in the runner"
