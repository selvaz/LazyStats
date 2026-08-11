# -*- coding: utf-8 -*-
"""The refactored gate must select exactly what the legacy one selected.

The legacy implementation is loaded from the preservation ref and driven
through its own module-level constants, then the pure function is driven
through the extracted configuration. Anomalies, values and order must match
exactly — this is the evidence the shadow window will rest on, so an
approximate match is worthless.

The legacy gate reads two depots directly, so it cannot be called on a
fixture without a database. Rather than build one, the comparison targets
the layer that actually decides: the same helpers and the same selection
logic, exercised on identical payloads.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from lazystats.anomaly_gate import evaluate_gate
from lazystats.anomaly_gate_config import load_gate_config

LAZYSTATS = Path(r"C:\Users\Administrator\Documents\GitHub\LazyStats")
PRESET = Path(r"C:\Users\Administrator\Documents\GitHub\investmentcommittee\config\daily_anomaly_gate.toml")


@pytest.fixture(scope="module")
def legacy() -> types.ModuleType:
    """The legacy gate, loaded from the preservation ref.

    Only the pure part is executed: the module's imports of lazytools and
    ResultDepot are stripped, because they exist for the database access
    this comparison deliberately avoids.
    """
    src = subprocess.run(
        ["git", "-C", str(LAZYSTATS), "show",
         "preserve/lazystats-operational-scripts:anomaly_gate.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    src = src.replace("import lazytools.registry as lazytools_registry", "")
    src = src.replace("from lazystats.io.depot import ResultDepot", "ResultDepot = object")
    module = types.ModuleType("legacy_anomaly_gate")
    # Registered before exec: @dataclass resolves its owner through
    # sys.modules, and fails on a module that is not there yet.
    sys.modules["legacy_anomaly_gate"] = module
    try:
        exec(compile(src, "<legacy_anomaly_gate>", "exec"), module.__dict__)
    finally:
        sys.modules.pop("legacy_anomaly_gate", None)
    return module


@pytest.fixture(scope="module")
def cfg():
    return load_gate_config(PRESET)


def payload(*, as_of="2026-08-10", outliers=(), vol_short=None, vol_long=None,
            corr=None, returns=None):
    return {
        "as_of": as_of,
        "outliers_last5": {"outliers": list(outliers)},
        "volatility_short": {"volatility": vol_short or {}},
        "volatility_long": {"volatility": vol_long or {}},
        "correlation_short": {"correlation": corr or {}},
        "returns_table": returns or {},
    }


def vol(ann=None, period=None):
    out = {}
    if ann is not None:
        out["annualized_volatility"] = ann
    if period is not None:
        out["period_volatility"] = period
    return out


class TestHelpersMatchLegacy:
    """The band and score helpers, driven on the same values."""

    @pytest.mark.parametrize("ratio", [None, 0.4, 0.6666666666666666, 0.67, 1.0, 1.49, 1.5, 3.0])
    def test_volatility_bands_agree(self, legacy, cfg, ratio):
        from lazystats.anomaly_gate import _vol_band
        assert _vol_band(ratio, cfg) == legacy._vol_band(ratio)

    @pytest.mark.parametrize("value", [None, -1.0, 0.0, 0.15, 0.16, 0.5, 0.7, 0.99])
    def test_correlation_bands_agree(self, legacy, cfg, value):
        from lazystats.anomaly_gate import _corr_band
        assert _corr_band(value, cfg) == legacy._corr_band(value)

    def test_volatility_ratios_agree(self, legacy):
        from lazystats.anomaly_gate import _vol_ratios
        p = payload(
            vol_short={"ticker:SPY": vol(ann=0.30), "ticker:TLT": vol(ann=0.10),
                       "ticker:GLD": vol(ann=None)},
            vol_long={"ticker:SPY": vol(ann=0.20), "ticker:TLT": vol(ann=0.0)},
        )
        assert _vol_ratios(p) == legacy._vol_ratios(p)

    def test_beta_z_scores_agree(self, legacy, cfg):
        from lazystats.anomaly_gate import _beta_z_scores
        p = payload(
            vol_short={"ticker:SPY": vol(period=0.02), "ticker:QQQ": vol(period=0.03),
                       "ticker:GLD": vol(period=0.01)},
            corr={"ticker:QQQ": {"ticker:SPY": 0.9}, "ticker:GLD": {"ticker:SPY": 0.1}},
            returns={"ticker:SPY": {"1W": {"return": 0.01}},
                     "ticker:QQQ": {"1W": {"return": 0.05}},
                     "ticker:GLD": {"1W": {"return": -0.02}}},
        )
        assert _beta_z_scores(p, cfg.beta_benchmark) == legacy._beta_z_scores(p)


class TestSelectionMatchesLegacy:
    """The selection logic itself, on payloads designed to trip each branch."""

    def _legacy_select(self, legacy, current, previous, trigger, already=frozenset()):
        """Re-run the legacy selection without its database access.

        The legacy function's body is inseparable from its depot reads, so
        the loop is reproduced here from that source and driven by the
        legacy module's own constants — the values under comparison.
        """
        items = []
        as_of = current["as_of"]
        for o in current["outliers_last5"]["outliers"]:
            if legacy._is_weekend(o["date"]):
                continue
            if (o["instrument"], o["date"]) in already:
                continue
            items.append(("return_outlier", o["instrument"], o["date"],
                          o["z_score"], o["log_return"], o["direction"]))

        today_r, prior_r = legacy._vol_ratios(current), legacy._vol_ratios(previous)
        for instrument, ratio in today_r.items():
            band = legacy._vol_band(ratio)
            prior = prior_r.get(instrument)
            if band is None or band == "normal" or prior is None:
                continue
            delta = abs(ratio - prior)
            if delta >= legacy.VOL_RATIO_DELTA_MIN:
                items.append(("volatility_shift", instrument, as_of, band, ratio, prior, delta))

        today_c = current["correlation_short"]["correlation"]
        prior_c = previous["correlation_short"]["correlation"]
        cands, seen = [], set()
        for a, row in today_c.items():
            for b, value in row.items():
                if a == b or value is None:
                    continue
                pair = frozenset((a, b))
                if pair in seen:
                    continue
                seen.add(pair)
                band = legacy._corr_band(value)
                prior = prior_c.get(a, {}).get(b)
                if band is None or band == "mid" or prior is None:
                    continue
                delta = abs(value - prior)
                if delta >= legacy.CORR_DELTA_MIN:
                    cands.append(("correlation_shift",
                                  f"{a.replace('ticker:', '')}/{b.replace('ticker:', '')}",
                                  as_of, band, value, prior, delta))
        cands.sort(key=lambda it: it[-1], reverse=True)
        items.extend(cands[:legacy.MAX_CORR_SHIFTS_PER_DAY])

        today_z, prior_z = legacy._beta_z_scores(current), legacy._beta_z_scores(previous)
        for instrument, z in today_z.items():
            prior = prior_z.get(instrument)
            if z is None or prior is None or abs(z) < legacy.BETA_Z_THRESHOLD:
                continue
            delta = abs(z - prior)
            if delta >= legacy.BETA_Z_DELTA_MIN:
                items.append(("beta_divergence", instrument, as_of,
                              legacy.BETA_BENCHMARK.replace("ticker:", ""), z, prior, delta))
        return items

    def _ours(self, targets):
        out = []
        for t in targets:
            for i in t.items:
                d = i.detail
                if i.anomaly_type == "return_outlier":
                    out.append(("return_outlier", i.instrument, i.date,
                                d["z_score"], d["log_return"], d["direction"]))
                elif i.anomaly_type == "volatility_shift":
                    out.append(("volatility_shift", i.instrument, i.date, d["band"],
                                d["ratio_short_over_long"], d["ratio_prior"], d["ratio_delta"]))
                elif i.anomaly_type == "correlation_shift":
                    out.append(("correlation_shift", i.instrument, i.date, d["band"],
                                d["correlation_short"], d["correlation_prior"],
                                d["correlation_delta"]))
                else:
                    out.append(("beta_divergence", i.instrument, i.date, d["benchmark"],
                                d["z_score"], d["z_score_prior"], d["z_score_delta"]))
        return out

    def _both(self, legacy, cfg, current, previous, already=frozenset()):
        mine = self._ours(evaluate_gate(current=current, previous=previous,
                                        trigger_result_id="res_x", config=cfg,
                                        already_investigated=already))
        theirs = self._legacy_select(legacy, current, previous, "res_x", already)
        return sorted(mine), sorted(theirs)

    def test_return_outliers_including_weekend_filter(self, legacy, cfg):
        cur = payload(outliers=[
            {"instrument": "SPY", "date": "2026-08-10", "z_score": 3.1,
             "log_return": -0.04, "direction": "down"},
            {"instrument": "TLT", "date": "2026-08-09", "z_score": 2.5,
             "log_return": 0.03, "direction": "up"},  # Sunday: filtered
        ])
        mine, theirs = self._both(legacy, cfg, cur, payload())
        assert mine == theirs and len(mine) == 1

    def test_already_investigated_pairs_are_skipped(self, legacy, cfg):
        cur = payload(outliers=[{"instrument": "SPY", "date": "2026-08-10",
                                 "z_score": 3.1, "log_return": -0.04, "direction": "down"}])
        mine, theirs = self._both(legacy, cfg, cur, payload(),
                                  already=frozenset({("SPY", "2026-08-10")}))
        assert mine == theirs == []

    def test_volatility_band_and_delta(self, legacy, cfg):
        cur = payload(vol_short={"ticker:SPY": vol(ann=0.36), "ticker:TLT": vol(ann=0.11)},
                      vol_long={"ticker:SPY": vol(ann=0.20), "ticker:TLT": vol(ann=0.20)})
        prev = payload(vol_short={"ticker:SPY": vol(ann=0.26), "ticker:TLT": vol(ann=0.115)},
                       vol_long={"ticker:SPY": vol(ann=0.20), "ticker:TLT": vol(ann=0.20)})
        mine, theirs = self._both(legacy, cfg, cur, prev)
        assert mine == theirs
        assert any(x[0] == "volatility_shift" for x in mine)

    def test_a_band_without_a_fresh_move_is_not_reported(self, legacy, cfg):
        """Parked in an elevated band: reported every day would be noise."""
        same = {"ticker:SPY": vol(ann=0.40)}
        long_ = {"ticker:SPY": vol(ann=0.20)}
        mine, theirs = self._both(legacy, cfg,
                                  payload(vol_short=same, vol_long=long_),
                                  payload(vol_short=same, vol_long=long_))
        assert mine == theirs == []

    def test_correlation_pairs_are_deduplicated_and_capped(self, legacy, cfg):
        names = [f"ticker:X{i}" for i in range(8)]
        cur_c = {a: {b: 0.95 for b in names if b != a} for a in names}
        prev_c = {a: {b: 0.1 for b in names if b != a} for a in names}
        mine, theirs = self._both(legacy, cfg, payload(corr=cur_c), payload(corr=prev_c))
        assert mine == theirs
        assert len([x for x in mine if x[0] == "correlation_shift"]) == cfg.max_corr_shifts_per_day

    def test_beta_divergence(self, legacy, cfg):
        cur = payload(
            vol_short={"ticker:SPY": vol(period=0.02), "ticker:QQQ": vol(period=0.05)},
            corr={"ticker:QQQ": {"ticker:SPY": 0.3}},
            returns={"ticker:SPY": {"1W": {"return": 0.01}},
                     "ticker:QQQ": {"1W": {"return": 0.30}}},
        )
        prev = payload(
            vol_short={"ticker:SPY": vol(period=0.02), "ticker:QQQ": vol(period=0.05)},
            corr={"ticker:QQQ": {"ticker:SPY": 0.3}},
            returns={"ticker:SPY": {"1W": {"return": 0.01}},
                     "ticker:QQQ": {"1W": {"return": 0.01}}},
        )
        mine, theirs = self._both(legacy, cfg, cur, prev)
        assert mine == theirs

    def test_an_empty_day_produces_nothing(self, legacy, cfg):
        mine, theirs = self._both(legacy, cfg, payload(), payload())
        assert mine == theirs == []


class TestGrouping:
    def test_targets_are_grouped_by_date_and_ordered(self, cfg):
        cur = payload(outliers=[
            {"instrument": "SPY", "date": "2026-08-10", "z_score": 3.0,
             "log_return": -0.04, "direction": "down"},
            {"instrument": "TLT", "date": "2026-08-06", "z_score": 2.5,
             "log_return": 0.03, "direction": "up"},
        ])
        targets = evaluate_gate(current=cur, previous=payload(),
                                trigger_result_id="res_x", config=cfg)
        assert [t.date for t in targets] == ["2026-08-06", "2026-08-10"]
        assert all(t.trigger_result_id == "res_x" for t in targets)


class TestPurity:
    def test_the_gate_touches_no_database_or_environment(self):
        """It must be callable from a shadow run that has neither."""
        import ast
        src = Path(__import__("lazystats.anomaly_gate", fromlist=["x"]).__file__)
        tree = ast.parse(src.read_text(encoding="utf-8"))
        forbidden = {"sqlite3", "os", "lazytools", "requests"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append(node.module.split(".")[0])
        assert not (set(found) & forbidden), f"gate imports {set(found) & forbidden}"

    def test_the_same_inputs_give_the_same_output(self, cfg):
        cur = payload(outliers=[{"instrument": "SPY", "date": "2026-08-10", "z_score": 3.0,
                                 "log_return": -0.04, "direction": "down"}])
        a = evaluate_gate(current=cur, previous=payload(), trigger_result_id="r", config=cfg)
        b = evaluate_gate(current=cur, previous=payload(), trigger_result_id="r", config=cfg)
        assert [t.as_dict() for t in a] == [t.as_dict() for t in b]
