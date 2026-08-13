"""The daily run's stored record, and the page rendered from it.

This is the report that survives the run that made it: no images, so it fits in
the depot, and a saved row renders again years later without refitting. The
failures worth guarding are the ones that still produce a file — a symbol
dropped, a probability vector lost, a tier that disagrees with the other report.

Both reports are built from the same records here. Their predecessors kept two
parallel descriptions of one run and could disagree about the day without
anything failing.
"""
from __future__ import annotations

import json

from lazystats.regimes.daily_payload import REPORT_KIND, build_payload
from lazystats.regimes.daily_render import render_html
from lazystats.regimes.report import Revision, SymbolReport

STATES = (
    {"state": 0, "label": "calm", "annualized_mean_return": 0.05,
     "annualized_volatility": 0.10},
    {"state": 1, "label": "mid", "annualized_mean_return": 0.01,
     "annualized_volatility": 0.22},
    {"state": 2, "label": "wild", "annualized_mean_return": -0.20,
     "annualized_volatility": 0.35},
)

REVISION = Revision(trading_date="2026-08-11", old_state=0, new_state=1,
                    old_prob_high_vol=0.12, new_prob_high_vol=0.88,
                    old_estimation_date="2026-08-12",
                    new_estimation_date="2026-08-13")


def entry(symbol: str = "GLD", **overrides) -> SymbolReport:
    base = {
        "symbol": symbol, "name": "Gold", "n_states": 3, "current_state": 0,
        "current_label": "calm", "current_tier": "calm", "is_high_vol": False,
        "prob_high_vol": 0.04, "current_state_probs": (0.91, 0.05, 0.04),
        "states": STATES, "bic": -34513.044, "loglik": 17316.761,
        "data_start": "2010-01-04", "data_end": "2026-08-13", "n_obs": 4172,
    }
    base.update(overrides)
    return SymbolReport(**base)


def payload(entries, **overrides) -> dict:
    kwargs = {"as_of": "2026-08-13", "periods_per_year": 252,
              "source": "lazystats.regimes.estimation"}
    kwargs.update(overrides)
    return build_payload(entries, **kwargs)


def row(entries=None, **overrides) -> dict:
    return {"result_id": "res_abc123", "kind": REPORT_KIND,
            "produced_by": "scheduled:run_regime_daily", "cadence": "stable",
            "created_at": "2026-08-13T13:45:00",
            "payload": payload([entry()] if entries is None else entries, **overrides)}


class TestTheRecord:
    def test_a_fitted_symbol_carries_every_state_not_only_the_current_one(self):
        """A regime call at 51% and one at 99% look identical in a table of
        labels, and they are not the same reading."""
        record = payload([entry()])["symbols"][0]
        assert len(record["states"]) == 3
        assert record["current_state_probs"] == [0.91, 0.05, 0.04]

    def test_a_failed_symbol_is_kept_with_its_reason(self):
        out = payload([entry(), SymbolReport(symbol="TLT", error="hub unreachable")])
        assert [e["symbol"] for e in out["errors"]] == ["TLT"]
        assert out["errors"][0]["error_msg"] == "hub unreachable"
        assert [s["symbol"] for s in out["symbols"]] == ["GLD"]

    def test_the_counts_follow_the_records(self):
        out = payload([entry("AAA", changed_today=True),
                       entry("BBB", revisions=(REVISION,)),
                       entry("CCC"),
                       SymbolReport(symbol="DDD", error="boom")])
        assert out["summary"] == {"n_ok": 3, "n_errors": 1,
                                  "n_changed_today": 1, "n_revised": 1}

    def test_a_revision_keeps_the_dates_behind_the_count(self):
        """A count alone cannot be checked against anything."""
        record = payload([entry(revisions=(REVISION,))])["symbols"][0]
        assert record["revised"] == 1
        assert record["revised_dates"] == ["2026-08-11"]

    def test_each_state_carries_its_tier(self):
        """Ranked once, in Python. Its predecessor ranked them again in the
        page's script, and the two could disagree about what counts as mid."""
        record = payload([entry()])["symbols"][0]
        assert [s["tier"] for s in record["states"]] == ["calm", "mid", "high"]

    def test_it_survives_a_round_trip_through_json(self):
        out = payload([entry(), SymbolReport(symbol="TLT", error="boom")])
        assert json.loads(json.dumps(out)) == out


class TestThePage:
    def test_the_placeholder_is_replaced(self):
        assert "__ROW_JSON__" not in render_html(row())

    def test_the_record_is_embedded_verbatim(self):
        source = row()
        assert json.dumps(source["payload"]["summary"]) in render_html(source)

    def test_a_closing_tag_inside_the_data_cannot_end_the_script(self):
        page = render_html(row([entry(), SymbolReport(symbol="X",
                                                      error="</script>boom")]))
        assert "</script>boom" not in page
        assert "<\\/script>boom" in page

    def test_the_filters_the_report_is_read_through_are_present(self):
        page = render_html(row())
        for view in ("changed", "revised", "highvol"):
            assert f'data-v="{view}"' in page

    def test_nothing_ranks_the_states_a_second_time_in_the_page(self):
        """The duplicated ranking its predecessor carried in JavaScript. If it
        comes back, the two reports can disagree about the same fit."""
        assert "annualized_volatility -" not in render_html(row())
        assert "volTiers" not in render_html(row())
