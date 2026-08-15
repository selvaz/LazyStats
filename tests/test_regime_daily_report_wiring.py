"""The daily runner's two pieces of report logic, against a real depot.

Neither needs the hub, so both run everywhere: what is exercised is how a fit
becomes a report record and how a changed date becomes a *revision*. The fitting
itself is the part that needs prices, and it is not what these check.

The distinction that matters: a date being stored for the first time is new, and
a date that had already been read and now reads differently is a revision.
Getting it wrong in either direction ruins the section — flag every symbol every
day and the real ones are invisible; suppress by position and a real one is
dropped on any day the market did not open.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("lazybridge")

from lazystats.io.depot import ResultDepot  # noqa: E402
from lazystats.regimes.config import RegimeConfig  # noqa: E402
from lazystats.regimes.persist import write_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_regime_daily.py"

KEY = "regime:GLD"
DATES = ["2026-08-11", "2026-08-12", "2026-08-13"]

STATES = [
    {"state": 0, "label": "calm", "annualized_mean_return": 0.05,
     "annualized_volatility": 0.10},
    {"state": 1, "label": "wild", "annualized_mean_return": -0.20,
     "annualized_volatility": 0.35},
]


def load_runner():
    spec = importlib.util.spec_from_file_location("_daily_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def config(**names) -> RegimeConfig:
    return RegimeConfig(instruments=("GLD",), windows=(), comparisons=(), s_max=3,
                        n_starts=20, random_state=123, retro_days=30,
                        names=dict(names))


def diagnostics(**overrides) -> dict:
    base = {"n_states": 2, "criterion": "bic", "bic": -34513.044,
            "loglik": 17316.761, "n_obs": len(DATES), "labels": ["calm", "wild"],
            "data_start": DATES[0], "data_end": DATES[-1], "periods_per_year": 252,
            "states": STATES, "transmat": [[0.9, 0.1], [0.2, 0.8]]}
    base.update(overrides)
    return base


def reading(state: int, prob: float = 0.1) -> dict:
    return {"state": state, "n_states": 2, "is_high_vol": state == 1,
            "prob_high_vol": prob}


def fitted(current: int = 0, chart: str | None = None) -> dict:
    readings = [reading(0), reading(0), reading(current)]
    return {"symbol": "GLD", "chart": chart, "dates": DATES,
            "readings": readings, "diagnostics": diagnostics()}


class TestARevisionIsAnOldDateThatMoved:
    @pytest.fixture
    def depot(self, tmp_path):
        return ResultDepot(str(tmp_path / "depot.sqlite"))

    def test_a_brand_new_trading_date_is_new_rather_than_revised(self, depot):
        """It writes on every single run. Counted as a revision, every symbol
        would be flagged every day and the real ones would be invisible."""
        runner = load_runner()
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-12",
                  diagnostics=diagnostics(), dates=DATES[:2],
                  readings=[reading(0)] * 2, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-13", diagnostics=diagnostics(),
                        dates=DATES, readings=[reading(0), reading(0), reading(1)],
                        retro_days=30)

        assert DATES[-1] in out.changed_dates
        assert runner._revisions_for(depot, KEY, out.changed_dates) == ()

    def test_a_revision_to_the_newest_date_is_not_lost_on_a_day_with_no_new_data(
            self, depot):
        """The market did not open, so the run's newest trading date is one
        already stored. Excluding the newest date -- as its predecessor did --
        drops a genuine revision here, and only here, which is why the rule is
        the prior vintage rather than the position."""
        runner = load_runner()
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-13",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 3, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-14", diagnostics=diagnostics(),
                        dates=DATES,
                        readings=[reading(0), reading(0), reading(1, 0.91)],
                        retro_days=30)

        revisions = runner._revisions_for(depot, KEY, out.changed_dates)
        assert [r.trading_date for r in revisions] == [DATES[-1]]
        assert (revisions[0].old_state, revisions[0].new_state) == (0, 1)

    def test_an_older_date_that_moved_is_reported_with_both_readings(self, depot):
        runner = load_runner()
        write_fit(depot, symbol="GLD", series_key=KEY, estimation_date="2026-08-12",
                  diagnostics=diagnostics(), dates=DATES,
                  readings=[reading(0)] * 3, retro_days=0)
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-13", diagnostics=diagnostics(),
                        dates=DATES,
                        readings=[reading(0), reading(1, 0.87), reading(0)],
                        retro_days=30)

        revisions = runner._revisions_for(depot, KEY, out.changed_dates)
        assert len(revisions) == 1
        assert revisions[0].trading_date == DATES[1]
        assert (revisions[0].old_state, revisions[0].new_state) == (0, 1)
        assert revisions[0].old_estimation_date == "2026-08-12"
        assert revisions[0].new_estimation_date == "2026-08-13"

    def test_a_date_with_only_one_vintage_is_not_a_revision(self, depot):
        runner = load_runner()
        out = write_fit(depot, symbol="GLD", series_key=KEY,
                        estimation_date="2026-08-13", diagnostics=diagnostics(),
                        dates=DATES, readings=[reading(0)] * 3, retro_days=0)
        assert runner._revisions_for(depot, KEY, out.changed_dates) == ()


class TestAFitBecomesAReportRecord:
    def test_the_current_tier_is_ranked_from_the_stored_statistics(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(current=1), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.current_tier == "high"
        assert entry.current_label == "wild"

    def test_the_calm_state_ranks_calm(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(current=0), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.current_tier == "calm"

    def test_the_display_name_comes_from_the_preset(self):
        runner = load_runner()
        entry = runner._entry(config(GLD="Gold"), fitted(), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.name == "Gold"

    def test_a_symbol_the_preset_does_not_name_carries_none(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.name is None

    def test_the_chart_arrives_decoded(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(chart="aGVsbG8="), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.chart == b"hello"

    def test_a_run_without_charts_carries_none(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.chart is None

    def test_the_probability_shown_is_the_latest_one(self):
        runner = load_runner()
        record = fitted()
        record["readings"][-1]["prob_high_vol"] = 0.42
        entry = runner._entry(config(), record, symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.prob_high_vol == 0.42


class TestTheRunRecordCarriesTheFullReading:
    """The fields the browsable report needs, which the port had dropped."""

    def test_the_probability_vector_reaches_the_record(self):
        runner = load_runner()
        record = fitted()
        record["readings"][-1]["state_probs"] = [0.7, 0.3]
        entry = runner._entry(config(), record, symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.current_state_probs == (0.7, 0.3)

    def test_the_current_state_index_reaches_the_record(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(current=1), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.current_state == 1

    def test_the_high_vol_flag_reaches_the_record(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(current=1), symbol="GLD",
                              revisions=(), changed_today=False)
        assert entry.is_high_vol is True

    def test_the_data_span_reaches_the_record(self):
        runner = load_runner()
        entry = runner._entry(config(), fitted(), symbol="GLD",
                              revisions=(), changed_today=False)
        assert (entry.data_start, entry.data_end) == (DATES[0], DATES[-1])
        assert entry.n_obs == len(DATES)


class TestTelegramDelivery:
    """The daily report goes out. Its predecessor attached it every day -- the
    legacy log carries fifteen 'Sent Telegram summary + report attachment'
    lines -- and the ported runner shipped without the flag, so the report was
    written and never delivered. Nothing failed: that is the shape of it.
    """

    def test_the_flag_exists_and_is_off_by_default(self):
        runner = load_runner()
        args = runner.build_parser().parse_args([
            "--config", "c.toml", "--window", "full", "--depot", "d",
            "--market-db", "m", "--production-db", "m"])
        assert args.send is False

    def test_without_the_flag_nothing_is_delivered(self, tmp_path):
        runner = load_runner()
        report = tmp_path / "r.html"
        report.write_text("x", encoding="utf-8")
        assert runner.delivery(False, False, report) == (False, "")

    def test_the_flag_with_a_report_delivers(self, tmp_path):
        runner = load_runner()
        report = tmp_path / "r.html"
        report.write_text("x", encoding="utf-8")
        assert runner.delivery(True, False, report) == (True, "")

    def test_sending_without_a_report_directory_refuses(self, tmp_path):
        """--send with nothing to attach is a request that cannot be honoured.
        Reporting success would claim a delivery that never happened."""
        runner = load_runner()
        deliver, reason = runner.delivery(True, False, None)
        assert deliver is False
        assert "--report-dir" in reason

    def test_a_report_that_was_never_written_refuses_too(self, tmp_path):
        runner = load_runner()
        deliver, reason = runner.delivery(True, False, tmp_path / "absent.html")
        assert deliver is False and "--report-dir" in reason

    def test_a_dry_run_sends_nothing_and_is_not_an_error(self, tmp_path):
        """--dry-run writes nothing to the depot; delivering its report would
        publish a run that officially did not happen."""
        runner = load_runner()
        report = tmp_path / "r.html"
        report.write_text("x", encoding="utf-8")
        deliver, reason = runner.delivery(True, True, report)
        assert deliver is False
        assert not reason.startswith("--send")

    def test_an_unconfigured_telegram_is_reported_not_swallowed(self, tmp_path,
                                                               monkeypatch, capsys):
        runner = load_runner()
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        report = tmp_path / "hmm_regime_report_20260814.html"
        report.write_text("<html></html>", encoding="utf-8")
        code = runner.send_telegram(report, {"symbols": 1, "fitted": 1, "failed": 0,
                                             "points_written": 0},
                                    window="full", as_of=date(2026, 8, 14))
        assert code == 2
        assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err

    def test_it_attaches_the_chart_report_and_says_what_ran(self, tmp_path,
                                                            monkeypatch, capsys):
        """The chart report is what the predecessor attached. The browsable one
        stays on disk and in the depot, where it can be reopened."""
        runner = load_runner()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        report = tmp_path / "hmm_regime_report_20260814.html"
        report.write_bytes(b"<html>chart</html>")

        sent = {}

        class FakeClient:
            @classmethod
            def from_token(cls, token):
                sent["token"] = token
                return cls()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def send_message(self, *, chat_id, text):
                sent["text"] = text

            def send_document(self, *, chat_id, document, filename, caption):
                sent["document"] = document
                sent["filename"] = filename

        module = types.ModuleType("lazytools.connectors.telegram")
        module.TelegramClient = FakeClient
        package = types.ModuleType("lazytools")
        connectors = types.ModuleType("lazytools.connectors")
        monkeypatch.setitem(sys.modules, "lazytools", package)
        monkeypatch.setitem(sys.modules, "lazytools.connectors", connectors)
        monkeypatch.setitem(sys.modules, "lazytools.connectors.telegram", module)

        code = runner.send_telegram(report, {"symbols": 109, "fitted": 109,
                                             "failed": 0, "points_written": 4816},
                                    window="full", as_of=date(2026, 8, 14))
        assert code == 0
        assert sent["document"] == b"<html>chart</html>"
        assert sent["filename"] == "hmm_regime_report_20260814.html"
        assert "109" in sent["text"] and "2026-08-14" in sent["text"]


# --------------------------------------------------------------------- #
# What the plan hands back, when it hands back nothing useful.
#
# `Agent` returns instead of raising when a step gives up, so the runner
# reaches `json.loads` with an empty string and the run dies reporting
# `Expecting value: line 1 column 1` -- the parser's complaint, not the
# cause. On 14 August the eight-year fit stopped after 48 of 109 instruments
# and that line was the whole of what the log held: no symbol, no step, no
# reason. These check that the runner now says which of the two happened.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["", "   ", "\n\t "])
def test_an_empty_answer_is_reported_as_the_plan_giving_up(raw):
    runner = load_runner()
    with pytest.raises(runner.PlanFailure) as caught:
        runner.decode_bundle(raw)
    message = str(caught.value)
    assert "returned nothing" in message
    # The depot keeps what was written before the stop; a run that says
    # otherwise sends someone looking for corruption that is not there.
    assert "still there" in message


def test_a_non_json_answer_is_quoted_back():
    runner = load_runner()
    with pytest.raises(runner.PlanFailure) as caught:
        runner.decode_bundle("Traceback (most recent call last): ...")
    message = str(caught.value)
    assert "not the run's result" in message
    assert "Traceback" in message, "the answer itself is the evidence; it must survive"


def test_a_real_answer_is_returned_unchanged():
    runner = load_runner()
    assert runner.decode_bundle('{"summary": {"failed": 0}}') == {"summary": {"failed": 0}}


# --------------------------------------------------------------------- #
# The failure the plan did report.
#
# `Agent` hands back an `Envelope`; a step that gives up sets `error` on it.
# This runner went straight to `.text()`, and so did the guard added after
# 14 August -- both saw an empty payload and concluded the plan had given no
# reason, while the reason sat unread on the object they were holding. The
# ETF runner has read that field all along.
# --------------------------------------------------------------------- #


class _Error:
    def __init__(self, type_, message):
        self.type, self.message = type_, message


class _Envelope:
    def __init__(self, error=None, text=""):
        self.error, self._text = error, text

    def text(self):
        return self._text


def test_a_reported_failure_is_carried_out_of_the_envelope():
    runner = load_runner()
    failure = _Error("StepError", "fit_symbol: IBCL.DE ran out of memory")
    reported = runner.plan_error(_Envelope(failure))
    assert reported is not None
    assert "StepError" in reported, "the kind of failure is dropped"
    assert "IBCL.DE" in reported, "the message is dropped, which is the whole point"


def test_an_envelope_without_an_error_reports_nothing():
    runner = load_runner()
    assert runner.plan_error(_Envelope(None, '{"summary": {}}')) is None


def test_the_error_is_read_before_the_payload():
    """Order, from the syntax tree: `.text()` after the check, never before.

    Reading the payload first is not merely redundant -- it is how this was
    wrong twice. An empty payload produces a confident, generic message that
    stops anyone looking for the real one.
    """
    import ast

    source = RUNNER.read_text(encoding="utf-8")
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "plan_error"
    ]
    texts = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "text"
    ]
    assert calls, "main() never asks the envelope whether the plan failed"
    assert texts, "main() never reads the payload"
    assert min(calls) < min(texts), (
        "the payload is read before the error is checked, which is how an "
        "empty payload came to speak for a failure that had a message"
    )
