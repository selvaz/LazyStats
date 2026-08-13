"""Rendering a stored comparison as a page.

The renderer is a pure function of a depot row, which is what makes a saved
comparison re-openable years later. These tests check the two things that would
silently produce a blank or a lying page: that the row actually reaches the
script, and that the window names come from the payload rather than from
anything the renderer assumes about which side is longer.
"""
from __future__ import annotations

import json

from lazystats.regimes.window_comparison import build_payload
from lazystats.regimes.window_comparison_render import render_html

CALM_THEN_WILD = [
    {"state": 0, "annualized_volatility": 0.10, "annualized_mean_return": 0.05},
    {"state": 1, "annualized_volatility": 0.35, "annualized_mean_return": -0.20},
]


def fit(window: str, current: int):
    from lazystats.regimes.window_comparison import WindowFit

    return WindowFit(window=window, n_states=2, current_state=current,
                     states=tuple(CALM_THEN_WILD), as_of="2026-08-13",
                     data_start="2010-01-01")


def row(readings=None, **overrides) -> dict:
    kwargs = {"comparison": "full_vs_eight", "baseline_window": "full",
              "candidate_window": "8y", "as_of": "2026-08-13",
              "periods_per_year": 252, "source": "lazystats.regimes.estimation"}
    kwargs.update(overrides)
    if readings is None:
        readings = [("GLD", fit("full", 0), fit("8y", 1))]
    return {
        "result_id": "res_abc123", "kind": "regime_window_comparison",
        "produced_by": "scheduled:run_regime_window_comparison",
        "created_at": "2026-08-13T17:00:00",
        "payload": build_payload(readings, **kwargs),
    }


class TestThePageCarriesItsOwnData:
    def test_the_placeholder_is_replaced(self):
        """Left in place, the script would throw and every section would render
        empty — a page that looks like a run with nothing in it."""
        assert "__ROW_JSON__" not in render_html(row())

    def test_the_row_is_embedded_verbatim(self):
        source = row()
        html = render_html(source)
        assert json.dumps(source["payload"]["summary"]) in html

    def test_a_closing_tag_inside_the_data_cannot_end_the_script(self):
        """A symbol carrying '</script>' would otherwise terminate the block
        early and blank the page."""
        html = render_html(row([("</script>GLD", fit("full", 0), fit("8y", 0))]))
        assert "</script>GLD" not in html
        assert "<\\/script>GLD" in html


class TestNeitherSideIsPrivileged:
    def test_the_window_names_come_from_the_payload(self):
        html = render_html(row(baseline_window="three_years",
                               candidate_window="ten_years"))
        assert "three_years" in html
        assert "ten_years" in html

    def test_nothing_in_the_page_is_hardcoded_to_a_full_history(self):
        """Its predecessor labelled the columns 'full' and 'windowed'. Comparing
        two bounded windows then produced a page describing a comparison nobody
        ran.

        The whole page is searched, script included — the table headers are
        built in JavaScript, so checking only the static markup would miss
        exactly where the old labels lived. Nothing rendered here is named
        'full' or 'windowed', so either word appearing at all is the template
        supplying it.
        """
        html = render_html(row(comparison="three_vs_ten",
                               baseline_window="three_years",
                               candidate_window="ten_years",
                               readings=[("GLD", fit("three_years", 0),
                                          fit("ten_years", 1))]))
        assert "full" not in html
        assert "windowed" not in html
