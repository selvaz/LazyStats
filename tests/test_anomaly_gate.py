"""What the gate selects, on fixtures built for this repository alone.

Self-contained: made-up instruments, the example preset's thresholds, and
expected results written out by hand rather than captured from a run — a
golden recorded by running the code under test only pins whatever it did,
including whatever it did wrong.

Each fixture is arranged so the expected answer can be read off the
thresholds. The example preset says a volatility ratio is high at 2.0 and
needs a fresh move of 0.25; the "parked" case sits at 4.0 with a move of
0.1, so it must select nothing, and the reason is arithmetic rather than
observation.

The gate's agreement with the pre-extraction implementation is a different
question, answered by a verifier outside this repository: it needs both the
legacy checkout and the private preset, neither of which belongs in a public
test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lazystats.anomaly_gate import evaluate_gate
from lazystats.anomaly_gate_config import load_gate_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "daily_anomaly_gate.example.toml"

BENCHMARK = "ticker:EXAMPLE"


@pytest.fixture(scope="module")
def cfg():
    return load_gate_config(EXAMPLE)


def payload(*, as_of="2026-08-10", outliers=(), vol_short=None, vol_long=None,
            corr=None, returns=None) -> dict:
    return {
        "as_of": as_of,
        "outliers_last5": {"outliers": list(outliers)},
        "volatility_short": {"volatility": vol_short or {}},
        "volatility_long": {"volatility": vol_long or {}},
        "correlation_short": {"correlation": corr or {}},
        "returns_table": returns or {},
    }


def outlier(instrument, date, z=4.0, ret=-0.06, direction="down") -> dict:
    return {"instrument": instrument, "date": date, "z_score": z,
            "log_return": ret, "direction": direction}


def selected(targets) -> list[tuple[str, str, str]]:
    """(date, anomaly_type, instrument) for everything selected, in order."""
    return [(t.date, i.anomaly_type, i.instrument) for t in targets for i in t.items]


def run(cfg, current, previous, *, already=frozenset(), trigger="res_example"):
    return evaluate_gate(current=current, previous=previous,
                         trigger_result_id=trigger, config=cfg,
                         already_investigated=already)


class TestReturnOutliers:
    def test_a_quiet_day_selects_nothing(self, cfg):
        assert run(cfg, payload(), payload(as_of="2026-08-07")) == ()

    def test_an_outlier_is_selected(self, cfg):
        targets = run(cfg, payload(outliers=[outlier("ticker:AAA", "2026-08-10")]),
                      payload(as_of="2026-08-07"))
        assert selected(targets) == [("2026-08-10", "return_outlier", "ticker:AAA")]

    def test_a_weekend_outlier_is_dropped(self, cfg):
        """2026-08-08 is a Saturday. The upstream source writes a placeholder
        row for non-trading days, which surfaces as a spurious z-score."""
        targets = run(cfg, payload(outliers=[outlier("ticker:AAA", "2026-08-08")]),
                      payload(as_of="2026-08-07"))
        assert targets == ()

    def test_an_already_investigated_outlier_is_dropped(self, cfg):
        targets = run(cfg, payload(outliers=[outlier("ticker:AAA", "2026-08-10")]),
                      payload(as_of="2026-08-07"),
                      already=frozenset({("ticker:AAA", "2026-08-10")}))
        assert targets == ()

    def test_outliers_are_grouped_by_date_in_order(self, cfg):
        targets = run(cfg, payload(outliers=[
            outlier("ticker:AAA", "2026-08-10"),
            outlier("ticker:BBB", "2026-08-07"),
            outlier("ticker:CCC", "2026-08-10"),
        ]), payload(as_of="2026-08-07"))
        assert [t.date for t in targets] == ["2026-08-07", "2026-08-10"]
        assert selected(targets) == [
            ("2026-08-07", "return_outlier", "ticker:BBB"),
            ("2026-08-10", "return_outlier", "ticker:AAA"),
            ("2026-08-10", "return_outlier", "ticker:CCC"),
        ]


class TestVolatility:
    """The preset bands at 2.0 and 0.5, with a required fresh move of 0.25."""

    def vols(self, short, long_):
        return ({"ticker:AAA": {"annualized_volatility": short}},
                {"ticker:AAA": {"annualized_volatility": long_}})

    def test_above_the_band_with_a_fresh_move_is_selected(self, cfg):
        cs, cl = self.vols(0.40, 0.10)   # ratio 4.0
        ps, pl = self.vols(0.11, 0.10)   # ratio 1.1, move of 2.9
        targets = run(cfg, payload(vol_short=cs, vol_long=cl),
                      payload(as_of="2026-08-07", vol_short=ps, vol_long=pl))
        assert selected(targets) == [("2026-08-10", "volatility_shift", "ticker:AAA")]

    def test_parked_in_the_band_without_a_fresh_move_is_not(self, cfg):
        cs, cl = self.vols(0.40, 0.10)   # ratio 4.0
        ps, pl = self.vols(0.39, 0.10)   # ratio 3.9, move of 0.1 < 0.25
        assert run(cfg, payload(vol_short=cs, vol_long=cl),
                   payload(as_of="2026-08-07", vol_short=ps, vol_long=pl)) == ()

    def test_below_the_band_with_a_fresh_move_is_selected(self, cfg):
        cs, cl = self.vols(0.02, 0.10)   # ratio 0.2
        ps, pl = self.vols(0.09, 0.10)   # ratio 0.9, move of 0.7
        targets = run(cfg, payload(vol_short=cs, vol_long=cl),
                      payload(as_of="2026-08-07", vol_short=ps, vol_long=pl))
        assert selected(targets) == [("2026-08-10", "volatility_shift", "ticker:AAA")]

    def test_inside_the_band_is_never_selected(self, cfg):
        cs, cl = self.vols(0.10, 0.10)   # ratio 1.0, squarely normal
        ps, pl = self.vols(0.40, 0.10)   # a large move, but into normality
        assert run(cfg, payload(vol_short=cs, vol_long=cl),
                   payload(as_of="2026-08-07", vol_short=ps, vol_long=pl)) == ()

    def test_a_missing_prior_reading_selects_nothing(self, cfg):
        """Freshness cannot be measured without something to measure from."""
        cs, cl = self.vols(0.40, 0.10)
        assert run(cfg, payload(vol_short=cs, vol_long=cl),
                   payload(as_of="2026-08-07")) == ()


class TestCorrelation:
    """Bands at 0.8 and 0.1, a required move of 0.3, and a cap of 5 per day."""

    def test_a_break_is_selected(self, cfg):
        targets = run(cfg, payload(corr={"ticker:AAA": {"ticker:BBB": -0.6}}),
                      payload(as_of="2026-08-07",
                              corr={"ticker:AAA": {"ticker:BBB": 0.5}}))
        assert [i.anomaly_type for t in targets for i in t.items] == ["correlation_shift"]

    def test_a_pair_is_reported_once_not_twice(self, cfg):
        """The matrix carries both halves; a pair is one finding."""
        both = {"ticker:AAA": {"ticker:BBB": 0.95}, "ticker:BBB": {"ticker:AAA": 0.95}}
        prior = {"ticker:AAA": {"ticker:BBB": 0.30}, "ticker:BBB": {"ticker:AAA": 0.30}}
        targets = run(cfg, payload(corr=both), payload(as_of="2026-08-07", corr=prior))
        assert sum(len(t.items) for t in targets) == 1

    def test_the_diagonal_is_ignored(self, cfg):
        targets = run(cfg, payload(corr={"ticker:AAA": {"ticker:AAA": 1.0}}),
                      payload(as_of="2026-08-07",
                              corr={"ticker:AAA": {"ticker:AAA": 0.0}}))
        assert targets == ()

    def test_a_move_below_the_minimum_is_not_selected(self, cfg):
        targets = run(cfg, payload(corr={"ticker:AAA": {"ticker:BBB": 0.95}}),
                      payload(as_of="2026-08-07",
                              corr={"ticker:AAA": {"ticker:BBB": 0.85}}))
        assert targets == ()

    def test_the_pair_label_is_sorted_not_iteration_order(self, cfg):
        """Regression for D8 (ecosystem-cleanup/docs/deferred-fixes.md): the
        label used to be composed from whichever of `a`/`b` the dict
        iteration reached first, so the same pair could read "AAA/BBB" from
        one serialization and "BBB/AAA" from another that preserved a
        different key order -- same matrix, disagreeing label. The matrix
        below only has the BBB->AAA half (still legal: `run()` reads
        `row.items()`, it does not require both halves), so a label built
        from encounter order would read "BBB/AAA"; sorted, it must read
        "AAA/BBB" regardless.
        """
        targets = run(cfg, payload(corr={"ticker:BBB": {"ticker:AAA": -0.6}}),
                      payload(as_of="2026-08-07",
                              corr={"ticker:BBB": {"ticker:AAA": 0.5}}))
        assert selected(targets) == [("2026-08-10", "correlation_shift", "AAA/BBB")]

    def test_the_daily_cap_holds(self, cfg):
        """A data glitch can produce hundreds of pairs; the cap is what stops
        one from inflating the investigation."""
        current = {f"ticker:A{i}": {f"ticker:B{i}": 0.99} for i in range(9)}
        prior = {f"ticker:A{i}": {f"ticker:B{i}": 0.05} for i in range(9)}
        targets = run(cfg, payload(corr=current), payload(as_of="2026-08-07", corr=prior))
        assert sum(len(t.items) for t in targets) == cfg.max_corr_shifts_per_day == 5

    def test_the_cap_keeps_the_largest_moves(self, cfg):
        """Truncating arbitrarily would drop the day's most important pairs."""
        current = {f"ticker:A{i}": {f"ticker:B{i}": 0.99} for i in range(9)}
        prior = {f"ticker:A{i}": {f"ticker:B{i}": 0.99 - (i + 1) / 10} for i in range(9)}
        targets = run(cfg, payload(corr=current), payload(as_of="2026-08-07", corr=prior))
        kept = [i.instrument for t in targets for i in t.items]
        assert kept == ["A8/B8", "A7/B7", "A6/B6", "A5/B5", "A4/B4"]


class TestBetaDivergence:
    """Threshold 3.0 sigma, with a required change of 1.5 from the prior day.

    beta = rho * vol_a / vol_benchmark = 0.8 * 0.05 / 0.02 = 2.0
    expected = 2.0 * 0.01 = 0.02; residual vol = 0.05 * sqrt(1 - 0.64) = 0.03
    """

    def frame(self, actual_return):
        return payload(
            vol_short={BENCHMARK: {"period_volatility": 0.02},
                       "ticker:AAA": {"period_volatility": 0.05}},
            corr={"ticker:AAA": {BENCHMARK: 0.8}},
            returns={BENCHMARK: {"1W": {"return": 0.01}},
                     "ticker:AAA": {"1W": {"return": actual_return}}},
        )

    def test_a_divergence_past_the_threshold_is_selected(self, cfg):
        # residual -0.11 over 0.03 is -3.67 sigma; yesterday's was 0.
        current = self.frame(-0.09)
        previous = self.frame(0.02)
        previous["as_of"] = "2026-08-07"
        targets = run(cfg, current, previous)
        assert selected(targets) == [("2026-08-10", "beta_divergence", "ticker:AAA")]

    def test_a_divergence_that_has_not_moved_is_not_selected(self, cfg):
        """Sitting at 3.67 sigma two days running is a state, not an event."""
        current = self.frame(-0.09)
        previous = self.frame(-0.09)
        previous["as_of"] = "2026-08-07"
        assert run(cfg, current, previous) == ()

    def test_the_benchmark_itself_is_never_selected(self, cfg):
        current = self.frame(-0.09)
        previous = self.frame(0.02)
        previous["as_of"] = "2026-08-07"
        assert BENCHMARK not in [i.instrument for t in run(cfg, current, previous)
                                 for i in t.items]

    def test_nothing_is_selected_without_the_benchmark(self, cfg):
        """The whole check is relative to it; without it there is no measure."""
        current = self.frame(-0.09)
        del current["volatility_short"]["volatility"][BENCHMARK]
        previous = self.frame(0.02)
        previous["as_of"] = "2026-08-07"
        assert run(cfg, current, previous) == ()


class TestTheResultCarriesItsRun:
    def test_every_target_carries_the_trigger_id(self, cfg):
        targets = run(cfg, payload(outliers=[outlier("ticker:AAA", "2026-08-10")]),
                      payload(as_of="2026-08-07"), trigger="res_abc")
        assert {t.trigger_result_id for t in targets} == {"res_abc"}

    def test_repeated_evaluation_is_deterministic(self, cfg):
        current = payload(outliers=[outlier("ticker:AAA", "2026-08-10"),
                                    outlier("ticker:BBB", "2026-08-10")],
                          corr={"ticker:AAA": {"ticker:BBB": -0.6}})
        previous = payload(as_of="2026-08-07", corr={"ticker:AAA": {"ticker:BBB": 0.5}})
        assert selected(run(cfg, current, previous)) == selected(run(cfg, current, previous))


class TestPropertiesTheShadowRestsOn:
    def test_the_gate_module_imports_no_database_or_network(self):
        """Read off the imports, not by patching: the shadow plan must not be
        able to open a store or make a request even by accident."""
        import ast

        source = (Path(__file__).resolve().parents[1] / "src" / "lazystats"
                  / "anomaly_gate.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & {"sqlite3", "os", "lazytools", "requests", "lazybridge"})
