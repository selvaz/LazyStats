"""The deterministic half of the weekly review.

Everything up to and including the prompt is a pure function of two stores
and a preset, so all of it can be checked without a language model existing
anywhere in the process. That is the half a shadow comparison can compare
exactly; the answer itself cannot be compared byte for byte and this file
does not pretend otherwise.

Self-contained: the depots are small in-memory fakes, the preset is the
fictitious example shipped here, and every instrument is made up.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from lazystats.weekly_anomaly import (
    build_prompt,
    format_daily_block,
    format_outliers_block,
    gather_week,
    last_week_end,
    review,
    review_payload,
    reviewed_instruments,
    save_review,
)
from lazystats.weekly_anomaly_config import (
    WeeklyConfigError,
    WeeklyReviewConfig,
    load_weekly_config,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weekly_anomaly_review.example.toml"


class FakeDepot:
    """A store, not a database.

    Rows are given in the order ``list`` should return them — newest first,
    the way a depot orders by creation time — so a test states the ordering
    it depends on instead of relying on one.
    """

    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or []
        self.saved: list[dict] = []

    def list(self, *, produced_by=None, cadence=None, limit=50):
        out = [r for r in self.rows
               if (produced_by is None or r["produced_by"] == produced_by)
               and (cadence is None or r.get("cadence") == cadence)]
        return [{"result_id": r["result_id"]} for r in out[:limit]]

    def load(self, result_id):
        for r in self.rows:
            if r["result_id"] == result_id:
                return r
        return None

    def save(self, **kwargs):
        self.saved.append(kwargs)
        return f"res_saved_{len(self.saved)}"


@pytest.fixture(scope="module")
def cfg():
    return load_weekly_config(EXAMPLE)


def explanation_row(result_id, day, items, produced_by="example.daily_explainer"):
    return {"result_id": result_id, "produced_by": produced_by, "cadence": "stable",
            "payload": {"date": day, "items": items}}


def item(instrument="ticker:AAA", day="2026-08-10", anomaly_type="return_outlier",
         category="macro_data", confidence="high", explanation="A data release."):
    return {"instrument": instrument, "anomaly_type": anomaly_type, "date": day,
            "category": category, "confidence": confidence, "explanation": explanation}


def stats_row(as_of="2026-08-14", outliers=(), produced_by="scheduled:example_daily_stats"):
    return {"result_id": "res_stats", "produced_by": produced_by, "cadence": "stable",
            "payload": {"as_of": as_of,
                        "outliers_last5": {"outliers": list(outliers)}}}


def outlier(instrument="ticker:AAA", day="2026-08-13", z=3.25, direction="down"):
    return {"instrument": instrument, "date": day, "z_score": z, "direction": direction}


class TestWhereTheWeekStarts:
    def test_the_first_run_falls_back_to_the_lookback(self, cfg):
        """Nothing to start from, so a boundary has to be chosen — and it is
        chosen from the preset, not from a constant here."""
        week = gather_week(explanations_depot=FakeDepot(), stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert week["week_start"] == "2026-08-08"

    def test_afterwards_it_resumes_from_the_last_review(self, cfg):
        """Reading the boundary rather than assuming seven days is what stops
        a missed Saturday quietly dropping a week of explanations."""
        depot = FakeDepot([
            {"result_id": "res_last", "produced_by": "example.weekly_review",
             "cadence": "stable", "payload": {"week_end": "2026-07-25"}},
        ])
        week = gather_week(explanations_depot=depot, stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert week["week_start"] == "2026-07-25"

    def test_a_review_row_that_cannot_be_loaded_falls_back(self, cfg):
        depot = FakeDepot()
        depot.rows = [{"result_id": "res_ghost", "produced_by": "example.weekly_review",
                       "cadence": "stable", "payload": {"week_end": "x"}}]
        depot.load = lambda _rid: None  # the index knows it, the store does not
        assert last_week_end(depot, cfg) is None

    def test_the_boundary_is_exclusive(self, cfg):
        """A row dated exactly on the previous week_end belongs to that week
        and has already been reviewed. Including it would double-count."""
        depot = FakeDepot([
            {"result_id": "res_last", "produced_by": "example.weekly_review",
             "cadence": "stable", "payload": {"week_end": "2026-08-08"}},
            explanation_row("res_a", "2026-08-08", [item(day="2026-08-08")]),
            explanation_row("res_b", "2026-08-09", [item(day="2026-08-09")]),
        ])
        week = gather_week(explanations_depot=depot, stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert [i["date"] for i in week["daily_items"]] == ["2026-08-09"]


class TestWhatTheWeekContains:
    def test_items_carry_the_row_they_came_from(self, cfg):
        """A verdict has to be traceable to the explanation it judges."""
        depot = FakeDepot([explanation_row("res_a", "2026-08-12", [item()])])
        week = gather_week(explanations_depot=depot, stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert week["daily_items"][0]["source_result_id"] == "res_a"

    def test_rows_from_another_producer_are_ignored(self, cfg):
        depot = FakeDepot([
            explanation_row("res_a", "2026-08-12", [item()], produced_by="someone.else"),
        ])
        week = gather_week(explanations_depot=depot, stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert week["daily_items"] == []

    def test_the_scan_limit_bounds_the_read(self, cfg):
        rows = [explanation_row(f"res_{i}", "2026-08-12", [item()]) for i in range(10)]
        tight = WeeklyReviewConfig(**{**vars(cfg), "daily_scan_limit": 3})
        week = gather_week(explanations_depot=FakeDepot(rows), stats_depot=FakeDepot(),
                           config=tight, today=date(2026, 8, 15))
        assert len(week["daily_items"]) == 3

    def test_the_freshest_snapshot_sets_the_week_end(self, cfg):
        stats = FakeDepot([stats_row(as_of="2026-08-14", outliers=[outlier()])])
        week = gather_week(explanations_depot=FakeDepot(), stats_depot=stats,
                           config=cfg, today=date(2026, 8, 15))
        assert week["week_end"] == "2026-08-14"
        assert week["latest_as_of"] == "2026-08-14"
        assert week["latest_outliers"]["outliers"][0]["instrument"] == "ticker:AAA"

    def test_without_a_snapshot_the_week_ends_today(self, cfg):
        """Stated rather than silently absent: the review still has a week,
        it just has nothing fresh to check against."""
        week = gather_week(explanations_depot=FakeDepot(), stats_depot=FakeDepot(),
                           config=cfg, today=date(2026, 8, 15))
        assert week["week_end"] == "2026-08-15"
        assert week["latest_outliers"] is None

    def test_today_is_passed_in_not_read_from_the_clock(self, cfg):
        """Otherwise the same inputs would give different answers on
        different days, and no comparison could be repeated."""
        first = gather_week(explanations_depot=FakeDepot(), stats_depot=FakeDepot(),
                            config=cfg, today=date(2026, 1, 1))
        second = gather_week(explanations_depot=FakeDepot(), stats_depot=FakeDepot(),
                             config=cfg, today=date(2026, 1, 1))
        assert first == second
        assert first["week_start"] == "2025-12-25"


class TestTheLayoutTheAgentSees:
    def test_an_empty_week_is_stated_not_blank(self):
        assert format_daily_block([]) == "(none)"

    def test_no_outliers_is_stated_not_blank(self):
        assert format_outliers_block(None) == "(no outliers)"
        assert format_outliers_block({"outliers": []}) == "(no outliers)"

    def test_items_are_numbered_from_one(self):
        block = format_daily_block([item(), item(instrument="ticker:BBB")])
        assert block.startswith("1. ticker:AAA")
        assert "\n2. ticker:BBB" in block

    def test_outliers_lose_the_prefix_and_keep_two_decimals(self):
        assert format_outliers_block({"outliers": [outlier(z=3.256)]}) == (
            "- AAA 2026-08-13: z=3.26 (down)"
        )

    def test_the_prompt_states_the_universe_and_threshold_from_the_preset(self, cfg):
        week = {"daily_items": [item()], "latest_as_of": "2026-08-14",
                "latest_outliers": {"outliers": [outlier()]}}
        prompt = build_prompt(week, cfg)
        assert cfg.universe_description in prompt
        assert f"|z| >= {cfg.outlier_threshold}" in prompt
        assert "1 items" in prompt

    def test_the_prompt_is_deterministic(self, cfg):
        week = {"daily_items": [item()], "latest_as_of": "2026-08-14",
                "latest_outliers": {"outliers": [outlier()]}}
        assert build_prompt(week, cfg) == build_prompt(week, cfg)

    def test_a_missing_as_of_is_named_rather_than_left_empty(self, cfg):
        week = {"daily_items": [], "latest_as_of": None, "latest_outliers": None}
        assert "as of unknown" in build_prompt(week, cfg)


class TestTheAnswerIsInjectedNotBuilt:
    """No engine is constructed anywhere in this module."""

    class FakeEnvelope:
        def __init__(self, payload=None, error=None):
            self.payload, self.error = payload, error
            self.ok = error is None

    def test_the_agent_receives_the_prompt_and_its_answer_is_returned(self, cfg):
        from lazystats.weekly_anomaly import WeeklyReview, WeeklySynthesis

        answer = WeeklyReview(
            verifications=[],
            synthesis=WeeklySynthesis(narrative="Quiet.", new_trends=[],
                                      regime_confirmations=[], new_risks=[]))
        seen = {}

        def agent(prompt):
            seen["prompt"] = prompt
            return self.FakeEnvelope(payload=answer)

        week = {"daily_items": [item()], "latest_as_of": "2026-08-14",
                "latest_outliers": None}
        assert review(week, cfg, agent=agent) is answer
        assert seen["prompt"] == build_prompt(week, cfg)

    def test_a_failed_agent_raises_rather_than_returning_nothing(self, cfg):
        def agent(_prompt):
            return self.FakeEnvelope(error="out of turns")

        with pytest.raises(RuntimeError, match="out of turns"):
            review({"daily_items": [], "latest_as_of": None, "latest_outliers": None},
                   cfg, agent=agent)

    def test_the_module_imports_no_model_database_or_messenger(self):
        """Read off the imports: the deterministic half must be exercisable
        with no model anywhere in the process."""
        source = (ROOT / "src" / "lazystats" / "weekly_anomaly.py").read_text(
            encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & {"lazytools", "lazybridge", "lazycrawler",
                                "sqlite3", "os", "requests"})


class TestPersistence:
    class Answer:
        def __init__(self):
            from lazystats.weekly_anomaly import WeeklySynthesis
            self.verifications = []
            self.synthesis = WeeklySynthesis(narrative="n", new_trends=["t"],
                                             regime_confirmations=[], new_risks=[])

    def week(self):
        return {"week_start": "2026-08-08", "week_end": "2026-08-14",
                "daily_items": [item(), item(instrument="ticker:BBB")]}

    def test_the_payload_records_the_boundary_and_the_count(self):
        payload = review_payload(self.week(), self.Answer())
        assert payload["week_start"] == "2026-08-08"
        assert payload["week_end"] == "2026-08-14"
        assert payload["n_daily_items"] == 2

    def test_instruments_are_deduplicated_and_stripped(self):
        week = self.week()
        week["daily_items"].append(item(instrument="ticker:AAA"))
        assert reviewed_instruments(week) == ["AAA", "BBB"]

    def test_the_row_is_written_under_the_configured_identity(self, cfg):
        depot = FakeDepot()
        result_id = save_review(self.week(), self.Answer(), depot=depot, config=cfg)
        assert result_id == "res_saved_1"
        saved = depot.saved[0]
        assert saved["produced_by"] == cfg.weekly_produced_by
        assert saved["series_key"] == cfg.weekly_series_key
        assert saved["cadence"] == "stable"

    def test_the_parameters_are_recorded_with_the_result(self, cfg):
        """A result whose thresholds are unrecorded cannot be re-read later
        and understood."""
        depot = FakeDepot()
        save_review(self.week(), self.Answer(), depot=depot, config=cfg)
        assert depot.saved[0]["provenance"]["parameters"] == cfg.as_provenance()


class TestTheConfiguration:
    def test_the_example_loads(self, cfg):
        assert cfg.initial_lookback_days == 7
        assert cfg.explainer.web is False

    def test_the_example_names_no_real_series(self, cfg):
        """A broken or realistic example is what a new user copies."""
        for value in (cfg.daily_produced_by, cfg.weekly_produced_by,
                      cfg.upstream_produced_by):
            assert "example" in value

    @pytest.mark.parametrize("key", [
        "daily_produced_by", "daily_series_key", "weekly_produced_by",
        "weekly_series_key", "upstream_produced_by", "upstream_series_key",
        "initial_lookback_days", "daily_scan_limit", "universe_description",
        "outlier_threshold",
    ])
    def test_every_key_is_required(self, tmp_path, key):
        body = "\n".join(ln for ln in EXAMPLE.read_text(encoding="utf-8").splitlines()
                         if not ln.startswith(f"{key} ="))
        p = tmp_path / "c.toml"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(WeeklyConfigError, match=key):
            load_weekly_config(p)

    def test_the_explainer_table_is_required(self, tmp_path):
        text = EXAMPLE.read_text(encoding="utf-8")
        p = tmp_path / "c.toml"
        p.write_text(text[:text.index("[explainer]")], encoding="utf-8")
        with pytest.raises(WeeklyConfigError, match=r"\[explainer\]"):
            load_weekly_config(p)

    def test_a_weekly_identity_equal_to_the_daily_one_is_refused(self, tmp_path):
        """It would have the review reading its own output as an explanation
        to verify, and produce a plausible result about nothing."""
        text = EXAMPLE.read_text(encoding="utf-8").replace(
            'weekly_produced_by = "example.weekly_review"',
            'weekly_produced_by = "example.daily_explainer"')
        p = tmp_path / "c.toml"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(WeeklyConfigError, match="must differ"):
            load_weekly_config(p)

    def test_a_shared_series_key_is_refused(self, tmp_path):
        text = EXAMPLE.read_text(encoding="utf-8").replace(
            'weekly_series_key = "example_weekly_review"',
            'weekly_series_key = "example_explanations"')
        p = tmp_path / "c.toml"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(WeeklyConfigError, match="must differ"):
            load_weekly_config(p)

    def test_a_boolean_web_flag_is_required(self, tmp_path):
        text = EXAMPLE.read_text(encoding="utf-8").replace("web = false", 'web = "no"')
        p = tmp_path / "c.toml"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(WeeklyConfigError, match="web"):
            load_weekly_config(p)

    def test_a_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(WeeklyConfigError, match="not found"):
            load_weekly_config(tmp_path / "absent.toml")

    def test_provenance_covers_every_field_but_the_agent(self, cfg):
        """The explainer choices are excluded on purpose: they describe how
        the answer was obtained, not what the review looked at, and a shadow
        run has no agent at all."""
        recorded = set(cfg.as_provenance())
        assert recorded == set(vars(cfg)) - {"explainer"}
