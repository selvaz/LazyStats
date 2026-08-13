"""The daily regime report: one self-contained page, charts included.

The report that goes out every day and is the one attached to the Telegram
message. A single file with no external assets — the charts are embedded as
base64 PNGs — because it has to survive being sent as an attachment and opened
anywhere.

:func:`render_html` is a pure function of the records handed to it. It reads no
database, fits nothing, and never touches matplotlib: the charts arrive already
drawn (see :mod:`~lazystats.regimes.charts`) and the revisions arrive already
read. That is what keeps the page's markup testable on its own, and what allows
the fitted models to be released one at a time instead of all being held until
the report is assembled.

The per-state table reports *annualized* mean return and volatility. Its
predecessor printed the fit's raw per-period parameters, which are not
comparable between two symbols and mean nothing without the frequency beside
them; these are the same numbers the depot now stores and the window comparison
ranks on.
"""
from __future__ import annotations

import base64
import html
from dataclasses import dataclass, field
from datetime import date
from typing import Any

_CSS = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0;
       background: #0B0F14; color: #D7E1EA; }
header { padding: 16px 24px; background: #10161d; border-bottom: 1px solid #2B3440; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #9AA7B4; font-size: 13px; }
nav { padding: 10px 24px; background: #10161d; border-bottom: 1px solid #2B3440;
      display: flex; flex-wrap: wrap; gap: 6px; }
nav a { color: #7BDFF2; text-decoration: none; font-size: 12px; padding: 3px 8px;
        border: 1px solid #2B3440; border-radius: 10px; }
nav a.flag { border-color: #FAA916; color: #FAA916; }
main { padding: 20px 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }
th, td { border-bottom: 1px solid #2B3440; padding: 6px 10px; text-align: left; }
th { color: #9AA7B4; font-weight: 600; }
tr.flag td { color: #FAA916; }
tr.err td { color: #EE6352; }
section.symbol { border-top: 2px solid #2B3440; padding-top: 18px; margin-top: 18px; }
section.symbol h2 { margin-bottom: 4px; }
img { max-width: 100%; border-radius: 4px; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 11px;
         margin-left: 6px; }
.badge.high { background: #EE6352; color: #0B0F14; }
.badge.mid { background: #F4D35E; color: #0B0F14; }
.badge.calm { background: #2D7DD2; color: #0B0F14; }
.badge.single, .badge.unknown { background: #556270; color: #D7E1EA; }
.badge.rev { background: #FAA916; color: #0B0F14; }
"""

_JS = """
function show(id) {
  document.querySelectorAll('section.symbol').forEach(s => s.style.display = 'none');
  document.getElementById('landing').style.display = id ? 'none' : 'block';
  if (id) document.getElementById(id).style.display = 'block';
}
"""

_TIER_LABELS = {"high": "high-vol", "mid": "mid vol", "calm": "calm",
                "single": "single state", "unknown": "unknown"}


@dataclass(frozen=True)
class Revision:
    """One trading date whose regime call moved between two estimation runs.

    The finding the whole append-on-change design exists to surface: not that
    today looks different, but that *yesterday* now reads differently than it
    did yesterday.
    """

    trading_date: str
    old_state: int
    new_state: int
    old_prob_high_vol: float | None
    new_prob_high_vol: float | None
    old_estimation_date: str
    new_estimation_date: str


@dataclass(frozen=True)
class SymbolReport:
    """Everything the page says about one symbol.

    A symbol whose fit failed carries ``error`` and nothing else; it still
    appears, because a symbol that silently vanished from the report would be
    indistinguishable from one nobody asked for.
    """

    symbol: str
    name: str | None = None
    error: str | None = None
    n_states: int = 0
    current_state: int | None = None
    current_label: str | None = None
    current_tier: str = "unknown"
    is_high_vol: bool = False
    prob_high_vol: float | None = None
    #: The whole probability vector behind the current call, one entry per
    #: state. What separates a regime read with conviction from a coin flip.
    current_state_probs: tuple[float, ...] = ()
    changed_today: bool = False
    states: tuple[dict[str, Any], ...] = ()
    transmat: tuple[tuple[float, ...], ...] = ()
    bic: float | None = None
    loglik: float | None = None
    data_start: str | None = None
    data_end: str | None = None
    n_obs: int | None = None
    chart: bytes | None = None
    revisions: tuple[Revision, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def flagged(self) -> bool:
        return self.changed_today or bool(self.revisions)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _name_div(entry: SymbolReport) -> str:
    return f'<div class="sub">{_esc(entry.name)}</div>' if entry.name else ""


def _badge(tier: str) -> str:
    label = _TIER_LABELS.get(tier, tier)
    return f'<span class="badge {_esc(tier)}">{_esc(label)}</span>'


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _chart_img(entry: SymbolReport) -> str:
    if not entry.chart:
        return '<p class="sub">No chart for this run.</p>'
    encoded = base64.b64encode(entry.chart).decode("ascii")
    return (f'<img alt="{_esc(entry.symbol)} regimes" '
            f'src="data:image/png;base64,{encoded}"/>')


def _stats_table(entry: SymbolReport) -> str:
    rows = "".join(
        f"<tr><td>{_esc(state.get('state'))}</td><td>{_esc(state.get('label'))}</td>"
        f"<td>{_number(state.get('annualized_mean_return'), 4)}</td>"
        f"<td>{_number(state.get('annualized_volatility'), 4)}</td></tr>"
        for state in entry.states
    )
    transmat = "".join(
        "<tr>" + "".join(f"<td>{_number(p)}</td>" for p in row) + "</tr>"
        for row in entry.transmat
    )
    head = (f"<p>Model: BIC={_number(entry.bic, 1)} | "
            f"logLik={_number(entry.loglik, 1)} | states={_esc(entry.n_states)}</p>")
    table = ("<table><tr><th>State</th><th>Label</th><th>Ann. mean return</th>"
             f"<th>Ann. volatility</th></tr>{rows}</table>")
    matrix = f"<h3>Transition matrix</h3><table>{transmat}</table>" if transmat else ""
    return head + table + matrix


def _revision_table(entry: SymbolReport) -> str:
    if not entry.revisions:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(r.trading_date)}</td>"
        f"<td>{_esc(r.old_state)} &rarr; {_esc(r.new_state)}</td>"
        f"<td>{_number(r.old_prob_high_vol)} &rarr; {_number(r.new_prob_high_vol)}</td>"
        f"<td>{_esc(r.old_estimation_date)} &rarr; {_esc(r.new_estimation_date)}</td></tr>"
        for r in entry.revisions
    )
    return ("<h3>Retroactive revisions</h3>"
            "<table><tr><th>Trading date</th><th>State</th><th>P(high-vol)</th>"
            f"<th>Estimation date</th></tr>{rows}</table>")


def render_html(entries: list[SymbolReport], *, as_of: date, generated: str,
                window: str | None = None) -> str:
    """Render the daily report.

    Args:
        entries: One record per symbol, fitted or failed.
        as_of: The date the run describes.
        generated: When the page was produced, already formatted. Passed in
            rather than read from the clock, so rendering the same records twice
            produces the same bytes and a test can compare them.
        window: The window's name, shown when the run was not the plain one.

    Returns:
        A complete, self-contained page.
    """
    fitted = sorted((e for e in entries if e.ok), key=lambda e: e.symbol)
    failed = sorted((e for e in entries if not e.ok), key=lambda e: e.symbol)
    changed = [e for e in fitted if e.changed_today]
    revised = [e for e in fitted if e.revisions]

    nav = "".join(
        f'<a class="{"flag" if e.flagged else ""}" href="#" '
        f"onclick=\"show('sec-{_esc(e.symbol)}');return false;\">{_esc(e.symbol)}</a>"
        for e in fitted
    ) + '<a href="#" onclick="show(null);return false;">&#8629; Recap</a>'

    # Flagged symbols first: the page opens on what moved, not on the alphabet.
    recap_rows = []
    for entry in sorted(fitted, key=lambda e: (not e.flagged, e.symbol)):
        flags = ""
        if entry.changed_today:
            flags += ' <span class="badge rev">changed today</span>'
        if entry.revisions:
            flags += f' <span class="badge rev">{len(entry.revisions)} revised</span>'
        recap_rows.append(
            f'<tr class="{"flag" if entry.flagged else ""}">'
            f'<td><a href="#" onclick="show(\'sec-{_esc(entry.symbol)}\');return false;">'
            f"{_esc(entry.symbol)}</a>{_name_div(entry)}</td>"
            f"<td>{_esc(entry.current_label)} {_badge(entry.current_tier)}</td>"
            f"<td>{_number(entry.prob_high_vol)}</td><td>{_esc(entry.n_states)}</td>"
            f"<td>{flags}</td></tr>"
        )
    for entry in failed:
        recap_rows.append(
            f'<tr class="err"><td>{_esc(entry.symbol)}{_name_div(entry)}</td>'
            f'<td colspan="4">ERROR: {_esc(entry.error)}</td></tr>'
        )

    sections = "".join(
        f'<section class="symbol" id="sec-{_esc(e.symbol)}" style="display:none">'
        f"<h2>{_esc(e.symbol)} &mdash; {_esc(e.current_label)}</h2>{_name_div(e)}"
        f"{_chart_img(e)}{_stats_table(e)}{_revision_table(e)}</section>"
        for e in fitted
    )

    window_note = f" &middot; window {_esc(window)}" if window else ""
    body = f"""
<header>
  <h1>HMM Regime Monitor &mdash; {_esc(as_of.isoformat())}</h1>
  <div class="sub">{len(fitted)} symbols fitted &middot; {len(changed)} regime changes today &middot;
  {len(revised)} with retroactive revisions &middot; {len(failed)} errors{window_note} &middot;
  generated {_esc(generated)}</div>
</header>
<nav>{nav}</nav>
<main>
  <div id="landing">
    <table><tr><th>Symbol</th><th>Current regime</th><th>P(high-vol)</th><th>#States</th><th>Flags</th></tr>
    {"".join(recap_rows)}
    </table>
  </div>
  {sections}
</main>
"""
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f"<title>HMM Regime Monitor {_esc(as_of.isoformat())}</title>"
            f"<style>{_CSS}</style></head>"
            f"<body>{body}<script>{_JS}</script></body></html>")


__all__ = ["Revision", "SymbolReport", "render_html"]
