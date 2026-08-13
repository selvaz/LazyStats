"""Render a stored window comparison as one self-contained HTML page.

:func:`render_html` is a pure function of the exact dict
:meth:`lazystats.io.depot.ResultDepot.load` returns — no database access, no
refitting — so any saved row can be re-rendered from its own JSON alone, years
after the fit that produced it.

The page reads the *symmetric* record
:func:`~lazystats.regimes.window_comparison.build_payload` writes: a baseline
and a candidate, each named by the configuration. Its predecessor rendered a
privileged "full history" against a "windowed" side, with the window's tag baked
into the column headers; here the two sides are labelled from the payload, so
comparing three years against ten produces a page that reads correctly without
anything in this module knowing which side is longer.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["render_html"]

_TEMPLATE = r"""<title>Regime Window Comparison — depot artifact</title>
<style>
  :root {
    --bg: #F5F6FA; --surface: #FFFFFF; --surface-2: #ECEEF3;
    --ink: #12151C; --ink-soft: #4B5468; --ink-faint: #8890A0;
    --border: #DDE1E8; --accent: #4C6FA6; --accent-ink: #2E4468;
    --pos: #B24A2E; --neg: #2E6E8E; --mid: #B8860B; --mid-ink: #7A5A07;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(20,24,34,.04), 0 8px 24px -12px rgba(20,24,34,.12);
  }
  :root[data-theme="dark"] {
    --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
    --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
    --border: #262B38; --accent: #7C9BD0; --accent-ink: #B9CCE8;
    --pos: #D97B5C; --neg: #5DA0C4; --mid: #E0BB4A; --mid-ink: #F0D48A;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
      --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
      --border: #262B38; --accent: #7C9BD0; --accent-ink: #B9CCE8;
      --pos: #D97B5C; --neg: #5DA0C4; --mid: #E0BB4A; --mid-ink: #F0D48A;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", ui-sans-serif, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
  body { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }
  .mono { font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums; }

  header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px;
    padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
  .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 6px; }
  h1 { font-size: 26px; font-weight: 700; letter-spacing: -.01em; margin: 0; text-wrap: balance; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, max-content)); gap: 4px 26px; text-align: right; }
  .meta-item .k { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); display: block; }
  .meta-item .v { font-size: 13px; color: var(--ink-soft); }

  section { margin-bottom: 32px; }
  .section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  h2 { font-size: 15px; font-weight: 700; letter-spacing: .01em; margin: 0; }
  .section-note { font-size: 12.5px; color: var(--ink-faint); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); }
  .empty-note { padding: 18px 20px; color: var(--ink-faint); font-size: 13px; }

  .flag-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; padding: 14px; }
  .flag-card { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; position: relative; overflow: hidden; }
  .flag-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--pos); }
  .flag-top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .flag-ticker { font-weight: 700; font-size: 14px; }
  .flag-compare { display: flex; align-items: center; gap: 8px; font-size: 12px; flex-wrap: wrap; }
  .pill { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  .pill.calm { color: var(--neg); background: color-mix(in srgb, var(--neg) 14%, transparent); }
  .pill.mid { color: var(--mid-ink); background: color-mix(in srgb, var(--mid) 20%, transparent); }
  .pill.high, .pill.highvol { color: var(--pos); background: color-mix(in srgb, var(--pos) 14%, transparent); }
  .pill.single, .pill.unknown { color: var(--ink-faint); background: var(--surface-2); }
  .mode-note { font-size: 10px; color: var(--ink-faint); font-style: italic; }
  .arrow { color: var(--ink-faint); }
  .flag-detail { font-size: 11px; color: var(--ink-faint); }

  table.compare { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.compare th { text-align: right; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 700; padding: 10px 8px; border-bottom: 1px solid var(--border); }
  table.compare th:first-child, table.compare td:first-child { text-align: left; }
  table.compare td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border); }
  table.compare tr:last-child td { border-bottom: none; }
  table.compare tr.disagree { background: color-mix(in srgb, var(--pos) 6%, transparent); }
  .ticker-cell { font-weight: 700; }
  .faint-cell { font-size: 11px; color: var(--ink-faint); }
  .wrap-x { overflow-x: auto; padding: 4px 14px 14px; }

  footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 11.5px;
    color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 22px; }
  footer b { color: var(--ink-soft); font-weight: 600; }
  footer .rule { flex-basis: 100%; line-height: 1.5; }
</style>

<header>
  <div>
    <p class="eyebrow">lazystats depot &middot; scheduled artifact</p>
    <h1>Regime Window Comparison</h1>
  </div>
  <div class="meta-grid mono" id="meta-grid"></div>
</header>

<section>
  <div class="section-head">
    <h2>Structural changes</h2>
    <span class="section-note">The two windows classify the symbol differently -- the signal this report exists for.</span>
  </div>
  <div class="card" id="disagree-section"></div>
</section>

<section>
  <div class="section-head">
    <h2>Single-state flags</h2>
    <span class="section-note">One window (or both) found no distinguishable regime structure -- not compared.</span>
  </div>
  <div class="card" id="single-section"></div>
</section>

<section>
  <div class="section-head">
    <h2>All symbols</h2>
  </div>
  <div class="card wrap-x">
    <table class="compare" id="all-table"></table>
  </div>
</section>

<footer id="footer"></footer>

<script>
const ROW = __ROW_JSON__;
const P = ROW.payload;
const BASE = P.baseline_window;
const CAND = P.candidate_window;

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function pill(tier) {
  const label = { calm: "calm", mid: "mid vol", high: "high-vol", highvol: "high-vol",
                  single: "single", unknown: "unknown" }[tier] || tier;
  return `<span class="pill ${esc(tier)}">${esc(label)}</span>`;
}
function modeNote(mode) {
  return mode === "direct"
    ? `<span class="mode-note">same state count -- compared directly</span>`
    : `<span class="mode-note">state counts differ -- collapsed to calm/high-vol</span>`;
}
function sideBySide(c) {
  return `
    <div class="flag-compare">
      <span>${esc(BASE)} (${esc(c.baseline.n_states)}s)</span>${pill(c.baseline.current_tier)}
      <span class="arrow">&rarr;</span>
      <span>${esc(CAND)} (${esc(c.candidate.n_states)}s)</span>${pill(c.candidate.current_tier)}
    </div>`;
}
function withAgreement(value) {
  return P.symbols.filter(s => s.comparison.status === "ok" && s.comparison.agreement === value);
}

(function renderHeader() {
  const items = [
    ["As of", P.as_of],
    ["Comparison", P.comparison],
    ["Baseline", BASE],
    ["Candidate", CAND],
    ["Symbols", P.summary.compared],
    ["Disagree", P.summary.disagree],
    ["Single-state", P.summary.single_state],
    ["Missing", P.summary.missing],
  ];
  document.getElementById("meta-grid").innerHTML = items.map(([k, v]) =>
    `<div class="meta-item"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");
})();

(function renderDisagree() {
  const flagged = withAgreement("disagree");
  const el = document.getElementById("disagree-section");
  if (!flagged.length) {
    el.innerHTML = `<div class="empty-note">No structural disagreements -- every symbol is classified the same way by both windows.</div>`;
    return;
  }
  el.innerHTML = `<div class="flag-grid">` + flagged.map(s => {
    const c = s.comparison;
    return `
      <div class="flag-card">
        <div class="flag-top"><span class="flag-ticker">${esc(s.symbol)}</span></div>
        ${sideBySide(c)}
        <div class="flag-detail">${esc(BASE)} data from ${esc(c.baseline.data_start)} &middot; ${esc(CAND)} data from ${esc(c.candidate.data_start)} &middot; ${modeNote(c.comparison_mode)}</div>
      </div>`;
  }).join("") + `</div>`;
})();

(function renderSingle() {
  const flagged = withAgreement("single_state");
  const el = document.getElementById("single-section");
  if (!flagged.length) {
    el.innerHTML = `<div class="empty-note">No single-state windows.</div>`;
    return;
  }
  el.innerHTML = `<div class="flag-grid">` + flagged.map(s => `
      <div class="flag-card">
        <div class="flag-top"><span class="flag-ticker">${esc(s.symbol)}</span></div>
        ${sideBySide(s.comparison)}
      </div>`).join("") + `</div>`;
})();

(function renderAll() {
  const thead = `<tr><th>Ticker</th><th>${esc(BASE)} states</th><th>${esc(BASE)} tier</th>`
    + `<th>${esc(CAND)} states</th><th>${esc(CAND)} tier</th><th>Agreement</th><th>Mode</th></tr>`;
  const tbody = P.symbols.map(s => {
    const c = s.comparison;
    if (c.status === "missing") {
      return `<tr><td class="ticker-cell">${esc(s.symbol)}</td><td colspan="6" class="faint-cell">`
        + `missing (${esc(BASE)}=${c.baseline_available}, ${esc(CAND)}=${c.candidate_available})</td></tr>`;
    }
    const rowCls = c.agreement === "disagree" ? "disagree" : "";
    const verdict = c.agreement === "disagree" ? "&#9888; disagree"
      : c.agreement === "single_state" ? "single" : "agree";
    return `
      <tr class="${rowCls}">
        <td class="ticker-cell">${esc(s.symbol)}</td>
        <td>${esc(c.baseline.n_states)}</td>
        <td>${pill(c.baseline.current_tier)}</td>
        <td>${esc(c.candidate.n_states)}</td>
        <td>${pill(c.candidate.current_tier)}</td>
        <td>${verdict}</td>
        <td class="faint-cell">${c.comparison_mode === "direct" ? "direct" : "collapsed"}</td>
      </tr>`;
  }).join("");
  document.getElementById("all-table").innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
})();

(function renderFooter() {
  const p = P.provenance;
  document.getElementById("footer").innerHTML = `
    <span><b>Comparison</b> ${esc(p.comparison)}</span>
    <span><b>Baseline</b> ${esc(p.baseline_window)}</span>
    <span><b>Candidate</b> ${esc(p.candidate_window)}</span>
    <span><b>Periods/year</b> ${esc(p.periods_per_year)}</span>
    <span><b>Source</b> ${esc(p.source)}</span>
    <span><b>Saved</b> ${esc(ROW.created_at)}</span>
    <span class="rule"><b>Rule</b> ${esc(p.classification_rule)}</span>
  `;
})();
</script>
"""


def render_html(row: dict[str, Any]) -> str:
    """Render one saved comparison row as a self-contained HTML page.

    Args:
        row: The dict :meth:`lazystats.io.depot.ResultDepot.load` returns, whose
            ``payload`` is what
            :func:`~lazystats.regimes.window_comparison.build_payload` wrote.

    Returns:
        A complete page, with the row embedded as the JSON the script reads.
        Pure — no I/O of any kind.
    """
    # `</` cannot appear raw inside a <script> block: an instrument or window
    # named with one would end the script early and blank the page.
    embedded = json.dumps(row).replace("</", "<\\/")
    return _TEMPLATE.replace("__ROW_JSON__", embedded)
