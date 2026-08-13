"""The daily report page.

The report that goes out every day and is the one attached to the Telegram
message, so the failures worth guarding are the ones that would still produce a
file: a symbol that quietly vanishes, a revision that is not shown, a chart that
is not embedded. All of them look like a successful run.

Everything here is a pure function of records. No depot, no fit, no matplotlib.
"""
from __future__ import annotations

import base64
from datetime import date

from lazystats.regimes.report import Revision, SymbolReport, render_html

AS_OF = date(2026, 8, 13)

STATES = (
    {"state": 0, "label": "calm", "annualized_mean_return": 0.05,
     "annualized_volatility": 0.10},
    {"state": 1, "label": "wild", "annualized_mean_return": -0.20,
     "annualized_volatility": 0.35},
)

# A one-pixel PNG: real bytes, so the base64 in the page is real base64.
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def entry(symbol: str = "GLD", **overrides) -> SymbolReport:
    base = {
        "symbol": symbol, "name": "Gold", "n_states": 2, "current_label": "calm",
        "current_tier": "calm", "prob_high_vol": 0.1234, "states": STATES,
        "transmat": ((0.9, 0.1), (0.2, 0.8)), "bic": -34513.044,
        "loglik": 17316.761, "chart": PIXEL,
    }
    base.update(overrides)
    return SymbolReport(**base)


def render(entries, **overrides) -> str:
    kwargs = {"as_of": AS_OF, "generated": "2026-08-13 13:45"}
    kwargs.update(overrides)
    return render_html(entries, **kwargs)


class TestEverySymbolIsAccountedFor:
    def test_a_fitted_symbol_gets_a_row_and_a_section(self):
        page = render([entry()])
        assert "GLD" in page
        assert 'id="sec-GLD"' in page

    def test_a_failed_symbol_still_appears_with_its_reason(self):
        """A symbol that vanished from the page would be indistinguishable from
        one nobody asked for — which is how a broken feed goes unnoticed."""
        page = render([entry(), SymbolReport(symbol="TLT", error="hub unreachable")])
        assert "TLT" in page
        assert "hub unreachable" in page

    def test_a_failed_symbol_is_not_counted_as_fitted(self):
        page = render([entry(), SymbolReport(symbol="TLT", error="boom")])
        assert "1 symbols fitted" in page
        assert "1 errors" in page


class TestWhatTheReportExistsToSurface:
    def test_a_regime_change_today_is_flagged(self):
        assert "changed today" in render([entry(changed_today=True)])

    def test_a_retroactive_revision_is_shown_with_both_readings(self):
        page = render([entry(revisions=(Revision(
            trading_date="2026-08-11", old_state=0, new_state=1,
            old_prob_high_vol=0.12, new_prob_high_vol=0.88,
            old_estimation_date="2026-08-12", new_estimation_date="2026-08-13"),))])
        assert "Retroactive revisions" in page
        assert "2026-08-11" in page
        assert "0.120" in page and "0.880" in page

    def test_a_symbol_with_nothing_to_report_carries_no_revision_table(self):
        assert "Retroactive revisions" not in render([entry()])

    def test_what_moved_is_listed_before_what_did_not(self):
        """In the recap, which is what the page opens on. The nav above it stays
        alphabetical: it is an index, and an index that reorders daily is
        useless for finding a symbol."""
        page = render([entry("AAA"), entry("ZZZ", changed_today=True)])
        recap = page.split('id="landing"')[1].split("</table>")[0]
        assert recap.index("ZZZ") < recap.index("AAA")


class TestThePageStandsAlone:
    def test_the_chart_is_embedded_rather_than_linked(self):
        """It is sent as an attachment: anything it fetches will not be there."""
        page = render([entry()])
        assert f"data:image/png;base64,{base64.b64encode(PIXEL).decode()}" in page
        assert "<img" in page

    def test_a_symbol_without_a_chart_still_renders(self):
        page = render([entry(chart=None)])
        assert "No chart" in page
        assert "<img" not in page

    def test_the_model_diagnostics_are_on_the_page(self):
        page = render([entry()])
        assert "BIC=-34513.0" in page
        assert "logLik=17316.8" in page

    def test_the_state_table_reports_annualized_figures(self):
        """Its predecessor printed the raw per-period parameters, which are not
        comparable between two symbols and mean nothing without the frequency."""
        page = render([entry()])
        assert "Ann. volatility" in page
        assert "0.3500" in page

    def test_the_transition_matrix_is_shown(self):
        assert "Transition matrix" in render([entry()])


class TestNothingInjectsItselfIntoThePage:
    def test_a_hostile_name_is_escaped(self):
        page = render([entry(name="<script>alert(1)</script>")])
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_an_error_message_is_escaped(self):
        page = render([SymbolReport(symbol="TLT", error="<b>boom</b>")])
        assert "<b>boom</b>" not in page


class TestRenderingIsDeterministic:
    def test_the_same_records_produce_the_same_bytes(self):
        """The timestamp is passed in, not read from the clock: otherwise two
        renders of one run differ and nothing can be compared."""
        entries = [entry(), SymbolReport(symbol="TLT", error="boom")]
        assert render(entries) == render(entries)

    def test_the_window_is_named_when_the_run_was_not_the_plain_one(self):
        assert "window 8y" in render([entry()], window="8y")
