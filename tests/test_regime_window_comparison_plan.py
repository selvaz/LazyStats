"""The window comparison runner, end to end against a real depot.

Everything here writes to a depot in ``tmp_path``. Nothing may touch a
configured one.

What these tests are for: the pieces below the runner are each covered on their
own, and each was green while the chain as a whole could not run at all —
nothing could build a fit out of the depot, so the comparison had no input. So
these exercise the runner the way the scheduled job does, through its own
command line, and assert on what it actually wrote.

No market database is opened, and none exists here: a comparison reads fits, not
prices. The paths passed as ``--market-db`` and ``--production-db`` are only
there to decide which series the keys point at.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# The runner composes the pipeline as a lazybridge Plan, declared as the `plans`
# extra. Absent it, there is no runner to test.
pytest.importorskip("lazybridge")

from lazystats.io.depot import ResultDepot  # noqa: E402
from lazystats.regimes.persist import write_fit  # noqa: E402
from lazystats.regimes.series import series_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_regime_window_comparison.py"
EXAMPLE = ROOT / "examples" / "regime_daily.example.toml"

SYMBOLS = ["SPY", "TLT", "GLD"]
DATES = ["2026-08-11", "2026-08-12", "2026-08-13"]

CALM_THEN_WILD = [
    {"state": 0, "label": "calm", "annualized_mean_return": 0.05,
     "annualized_volatility": 0.10},
    {"state": 1, "label": "wild", "annualized_mean_return": -0.20,
     "annualized_volatility": 0.35},
]


def load_runner():
    """Import the root-level runner as a module.

    It is a script beside the package, not part of it, so pytest's ``pythonpath``
    does not cover it.
    """
    spec = importlib.util.spec_from_file_location("_comparison_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paths(tmp_path):
    """A depot, an output directory, and a market database path never opened."""
    market = tmp_path / "market_data.duckdb"
    market.write_bytes(b"")  # never read; only its path identifies the series
    return {
        "depot": str(tmp_path / "depot.sqlite"),
        "out": str(tmp_path / "reports"),
        "market": str(market),
    }


def seed(paths, symbol: str, *, variant: str | None, current: int,
         states=CALM_THEN_WILD) -> None:
    """One window's fit for one symbol, written the way the daily job writes it."""
    key = series_key(symbol, market_db=paths["market"],
                     production_db=paths["market"], variant=variant)
    readings = [{"state": 0, "n_states": len(states), "is_high_vol": False}] * 2
    readings.append({"state": current, "n_states": len(states),
                     "is_high_vol": current == 1})
    depot = ResultDepot(paths["depot"])
    try:
        write_fit(depot, symbol=symbol, series_key=key,
                  estimation_date="2026-08-13",
                  diagnostics={"n_states": len(states), "criterion": "bic",
                               "bic": 100.0, "loglik": -50.0, "n_obs": len(DATES),
                               "labels": [s["label"] for s in states],
                               "data_start": DATES[0], "data_end": DATES[-1],
                               "periods_per_year": 252, "states": states},
                  dates=DATES, readings=readings, retro_days=0)
    finally:
        depot.close()


def run(runner, paths, monkeypatch, *extra: str) -> int:
    argv = ["run_regime_window_comparison.py",
            "--config", str(EXAMPLE), "--comparison", "full_vs_8y",
            "--depot", paths["depot"], "--market-db", paths["market"],
            "--production-db", paths["market"], "--out-dir", paths["out"],
            "--as-of", "2026-08-13", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return runner.main()


def reports(paths) -> list[Path]:
    return sorted(Path(paths["out"]).glob("*.html"))


def saved_rows(paths) -> list[dict]:
    depot = ResultDepot(paths["depot"])
    try:
        return [depot.load(e["result_id"])
                for e in depot.list(cadence="stable", limit=5000)
                if e["kind"] == "regime_window_comparison"]
    finally:
        depot.close()


class TestAFullRun:
    @pytest.fixture(autouse=True)
    def both_windows_fitted(self, paths):
        for symbol in SYMBOLS:
            seed(paths, symbol, variant=None, current=0)
        # GLD ends in the turbulent state under the shorter window only: the one
        # structural disagreement this run should find.
        for symbol in SYMBOLS:
            seed(paths, symbol, variant="8y", current=1 if symbol == "GLD" else 0)

    def test_it_succeeds_and_writes_one_report(self, paths, monkeypatch, capsys):
        assert run(load_runner(), paths, monkeypatch) == 0
        assert len(reports(paths)) == 1

    def test_it_finds_the_disagreement_and_only_that_one(self, paths, monkeypatch,
                                                         capsys):
        run(load_runner(), paths, monkeypatch)
        summary = json.loads(capsys.readouterr().out)
        assert summary["compared"] == 3
        assert summary["disagree"] == 1
        assert summary["agree"] == 2
        assert summary["missing"] == 0

    def test_the_verdict_is_stored_where_it_can_be_read_again(self, paths,
                                                             monkeypatch, capsys):
        run(load_runner(), paths, monkeypatch)
        rows = saved_rows(paths)
        assert len(rows) == 1
        flagged = [s["symbol"] for s in rows[0]["payload"]["symbols"]
                   if s["comparison"].get("agreement") == "disagree"]
        assert flagged == ["GLD"]

    def test_the_stored_row_alone_renders_the_page(self, paths, monkeypatch, capsys):
        """The property the whole design rests on: a saved comparison is
        re-openable from its own JSON, with no depot and no refit."""
        from lazystats.regimes.window_comparison_render import render_html

        run(load_runner(), paths, monkeypatch)
        page = render_html(saved_rows(paths)[0])
        assert page == reports(paths)[0].read_text(encoding="utf-8")

    def test_the_report_is_named_for_the_result_it_shows(self, paths, monkeypatch,
                                                         capsys):
        """Two runs on the same day must not overwrite each other's report."""
        run(load_runner(), paths, monkeypatch)
        assert saved_rows(paths)[0]["result_id"] in reports(paths)[0].name


class TestADryRun:
    @pytest.fixture(autouse=True)
    def both_windows_fitted(self, paths):
        for symbol in SYMBOLS:
            seed(paths, symbol, variant=None, current=0)
            seed(paths, symbol, variant="8y", current=0)

    def test_it_writes_the_page_but_stores_nothing(self, paths, monkeypatch, capsys):
        assert run(load_runner(), paths, monkeypatch, "--dry-run") == 0
        assert len(reports(paths)) == 1
        assert saved_rows(paths) == []

    def test_the_page_says_it_was_not_saved(self, paths, monkeypatch, capsys):
        """Rendering an unsaved run under a result id nobody could look up would
        make a throwaway indistinguishable from the real thing."""
        run(load_runner(), paths, monkeypatch, "--dry-run")
        assert "dry run" in reports(paths)[0].read_text(encoding="utf-8")

    def test_send_is_ignored_on_a_dry_run(self, paths, monkeypatch, capsys):
        """--dry-run and --send together must not deliver: the guard is the
        combination, and it is the one a scheduled task gets wrong."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert run(load_runner(), paths, monkeypatch, "--dry-run", "--send") == 0


class TestWhatCannotBeCompared:
    def test_one_side_never_fitted_is_reported_as_missing(self, paths, monkeypatch,
                                                          capsys):
        for symbol in SYMBOLS:
            seed(paths, symbol, variant=None, current=0)
        seed(paths, "GLD", variant="8y", current=0)

        assert run(load_runner(), paths, monkeypatch) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["missing"] == 2
        assert summary["compared"] == 3

    def test_a_run_that_compared_nothing_fails(self, paths, monkeypatch, capsys):
        """Every symbol missing means an upstream window was never fitted. This
        is exactly how two scheduled jobs failed unnoticed for five days: the
        job that depended on them kept reporting success."""
        for symbol in SYMBOLS:
            seed(paths, symbol, variant=None, current=0)
        assert run(load_runner(), paths, monkeypatch) == 1


class TestTheConfigurationDecides:
    def test_an_undeclared_comparison_refuses_to_start(self, paths, monkeypatch,
                                                       capsys):
        runner = load_runner()
        monkeypatch.setattr(sys, "argv", [
            "run_regime_window_comparison.py", "--config", str(EXAMPLE),
            "--comparison", "not_declared", "--depot", paths["depot"],
            "--market-db", paths["market"], "--production-db", paths["market"],
            "--out-dir", paths["out"]])
        assert runner.main() == 2
        assert "not declared" in capsys.readouterr().err

    def test_a_missing_configuration_refuses_to_start(self, paths, monkeypatch,
                                                      capsys):
        """There is no default preset, and a job comparing an unstated universe
        is worse than one that will not start."""
        runner = load_runner()
        monkeypatch.setattr(sys, "argv", [
            "run_regime_window_comparison.py", "--config", "no_such_file.toml",
            "--comparison", "full_vs_8y", "--depot", paths["depot"],
            "--market-db", paths["market"], "--production-db", paths["market"],
            "--out-dir", paths["out"]])
        assert runner.main() == 2


class TestItNeverOpensTheMarketDatabase:
    def test_the_run_completes_with_an_unreadable_market_database(self, paths,
                                                                  monkeypatch,
                                                                  capsys):
        """The market path identifies the series; it is never opened. Pointing it
        at a file that is not a database at all proves the difference — and a
        comparison that needed the prices would fail here."""
        for symbol in SYMBOLS:
            seed(paths, symbol, variant=None, current=0)
            seed(paths, symbol, variant="8y", current=0)
        Path(paths["market"]).write_text("not a database", encoding="utf-8")
        assert run(load_runner(), paths, monkeypatch) == 0
