"""Render a stored daily run as one self-contained, browsable page.

:func:`render_html` is a pure function of the exact dict
:meth:`lazystats.io.depot.ResultDepot.load` returns — no database access, no
refitting — so any saved row can be re-rendered from its own JSON alone. That is
the difference between this report and the chart-based one beside it: this one
survives the run that made it.

Every state is shown for every symbol, not only the one it is in, with the
probability behind each. A regime call at 51% and a regime call at 99% look
identical in a table of labels, and they are not the same reading.

The tiers come from the payload. Its predecessor ranked the states again here,
in JavaScript, alongside the Python that ranked them for the other report — two
copies of "what counts as mid vol", with a comment hoping they would not drift.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["render_html"]

_TEMPLATE = r"""<title>HMM Regime Monitor — depot artifact</title>
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

  .filter-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  .filter-bar input { flex: 1; min-width: 160px; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface); color: var(--ink); font: inherit; font-size: 13px; }
  .toggle-group { display: inline-flex; background: var(--surface-2); border-radius: 999px; padding: 3px; gap: 2px; }
  .toggle-group button { border: none; background: transparent; font: inherit; font-size: 12px; font-weight: 600;
    color: var(--ink-soft); padding: 6px 14px; border-radius: 999px; cursor: pointer; }
  .toggle-group button.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }

  .symbol-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
  .symbol-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
  .symbol-card.flag { border-color: var(--accent); }
  .symbol-top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .symbol-ticker { font-weight: 700; font-size: 15px; }
  .symbol-name { font-size: 11px; color: var(--ink-faint); }
  .badge { font-size: 10px; font-weight: 700; letter-spacing: .02em; padding: 2px 7px; border-radius: 999px; }
  .badge.changed { color: var(--pos); background: color-mix(in srgb, var(--pos) 14%, transparent); }
  .badge.revised { color: var(--accent); background: color-mix(in srgb, var(--accent) 16%, transparent); }
  .badge.high { color: var(--pos); background: color-mix(in srgb, var(--pos) 14%, transparent); }
  .badge.mid { color: var(--mid-ink); background: color-mix(in srgb, var(--mid) 20%, transparent); }
  .badge.calm { color: var(--neg); background: color-mix(in srgb, var(--neg) 14%, transparent); }
  .badge.single, .badge.unknown { color: var(--ink-faint); background: var(--surface-2); }
  .badges { display: flex; gap: 5px; flex-wrap: wrap; }

  .state-row { display: grid; grid-template-columns: 20px 1fr 48px 64px 64px 48px; align-items: center; gap: 8px; font-size: 11.5px; }
  .state-row .lbl { color: var(--ink-soft); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .state-row.current .lbl { color: var(--ink); font-weight: 700; }
  .state-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--surface-2); }
  .state-row.current .state-dot { background: var(--accent); }
  .state-prob { text-align: right; font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; color: var(--ink-soft); }
  .state-row.current .state-prob { color: var(--ink); font-weight: 700; }
  .prob-track { position: relative; height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .prob-fill { position: absolute; inset: 0; background: var(--accent); border-radius: 4px; opacity: .35; }
  .state-row.current .prob-fill { opacity: 1; }
  .state-stat { text-align: right; font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .state-stat.pos { color: var(--pos); }
  .state-stat.neg { color: var(--neg); }
  .tier-tag { text-align: center; font-size: 9px; font-weight: 700; letter-spacing: .02em; padding: 1px 5px; border-radius: 999px; }
  .tier-tag.calm { color: var(--neg); background: color-mix(in srgb, var(--neg) 14%, transparent); }
  .tier-tag.mid { color: var(--mid-ink); background: color-mix(in srgb, var(--mid) 20%, transparent); }
  .tier-tag.high { color: var(--pos); background: color-mix(in srgb, var(--pos) 14%, transparent); }
  .tier-tag.single, .tier-tag.unknown { color: var(--ink-faint); background: var(--surface-2); }

  .fit-note { font-size: 10.5px; color: var(--ink-faint); }
  .col-head { display: grid; grid-template-columns: 20px 1fr 48px 64px 64px 48px; gap: 8px; font-size: 10px;
    color: var(--ink-faint); font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }
  .col-head span:nth-child(n+3) { text-align: right; }

  .error-card { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--pos);
    border-radius: var(--radius); padding: 10px 14px; font-size: 12.5px; margin-bottom: 8px; }

  footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 11.5px;
    color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 22px; }
  footer b { color: var(--ink-soft); font-weight: 600; }
</style>

<header>
  <div>
    <p class="eyebrow">lazystats depot &middot; scheduled artifact</p>
    <h1>HMM Regime Monitor</h1>
  </div>
  <div class="meta-grid mono" id="meta-grid"></div>
</header>

<div class="filter-bar">
  <input type="text" id="search" placeholder="Filter by ticker or name...">
  <div class="toggle-group" id="view-toggle">
    <button data-v="all" class="active">All</button>
    <button data-v="changed">Changed today</button>
    <button data-v="revised">Revised</button>
    <button data-v="highvol">High-vol now</button>
  </div>
</div>

<div class="symbol-grid" id="symbol-grid"></div>

<section style="margin-top:28px" id="errors-section"></section>

<footer id="footer"></footer>

<script>
const ROW = __ROW_JSON__;
const P = ROW.payload;

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function pct(v) { return v == null ? "&mdash;" : (v * 100).toFixed(1) + "%"; }
function fixed(v, d) { return v == null ? "&mdash;" : Number(v).toFixed(d); }

const TIER_LABEL = { calm: "calm", mid: "mid vol", high: "high-vol",
                     single: "single state", unknown: "unknown" };

let currentFilter = "all";
let currentQuery = "";

function matchesFilter(s) {
  if (currentFilter === "changed" && !s.changed_today) return false;
  if (currentFilter === "revised" && !s.revised) return false;
  if (currentFilter === "highvol" && !s.is_high_vol) return false;
  if (currentQuery) {
    const hay = (s.symbol + " " + (s.name || "")).toLowerCase();
    if (!hay.includes(currentQuery.toLowerCase())) return false;
  }
  return true;
}

function stateRow(s, st) {
  const prob = (s.current_state_probs && s.current_state_probs[st.state]) ?? 0;
  const isCurrent = st.state === s.current_state;
  const tier = st.tier || "unknown";
  return `
    <div class="state-row ${isCurrent ? "current" : ""}">
      <span class="state-dot"></span>
      <span class="lbl">${esc(st.label)}</span>
      <span class="state-prob">${(prob * 100).toFixed(1)}%</span>
      <span class="state-stat ${st.annualized_mean_return >= 0 ? "pos" : "neg"}">${pct(st.annualized_mean_return)}</span>
      <span class="state-stat">${pct(st.annualized_volatility)}</span>
      <span class="tier-tag ${esc(tier)}">${esc(tier)}</span>
    </div>
    <div class="prob-track" style="grid-column:1/-1;margin-bottom:4px;"><span class="prob-fill" style="width:${(prob*100).toFixed(1)}%"></span></div>`;
}

function renderSymbols() {
  const grid = document.getElementById("symbol-grid");
  const visible = P.symbols.filter(matchesFilter);
  if (!visible.length) {
    grid.innerHTML = `<div class="fit-note">No symbols match this filter.</div>`;
    return;
  }
  grid.innerHTML = visible.map(s => {
    const flagged = s.changed_today || s.revised > 0;
    const tier = s.current_tier || "unknown";
    const badges = [];
    if (s.changed_today) badges.push(`<span class="badge changed">changed today</span>`);
    if (s.revised) badges.push(`<span class="badge revised">${s.revised} revised</span>`);
    badges.push(`<span class="badge ${esc(tier)}">${esc(TIER_LABEL[tier] || tier)}</span>`);

    return `
      <div class="symbol-card ${flagged ? "flag" : ""}">
        <div class="symbol-top">
          <span><span class="symbol-ticker">${esc(s.symbol)}</span>${s.name ? ` <span class="symbol-name">${esc(s.name)}</span>` : ""}</span>
          <div class="badges">${badges.join("")}</div>
        </div>
        <div class="col-head">
          <span></span><span>State</span><span>Prob.</span><span>Ann. mean</span><span>Ann. vol</span><span></span>
        </div>
        ${s.states.map(st => stateRow(s, st)).join("")}
        <div class="fit-note">n_obs=${esc(s.fit.n_obs ?? "—")} &middot; data ${esc(s.fit.data_start ?? "?")} &rarr; ${esc(s.fit.data_end ?? "?")} &middot; BIC=${fixed(s.fit.bic, 1)}</div>
      </div>`;
  }).join("");
}

document.getElementById("view-toggle").addEventListener("click", e => {
  const btn = e.target.closest("button");
  if (!btn) return;
  currentFilter = btn.dataset.v;
  document.querySelectorAll("#view-toggle button").forEach(b => b.classList.toggle("active", b === btn));
  renderSymbols();
});
document.getElementById("search").addEventListener("input", e => {
  currentQuery = e.target.value;
  renderSymbols();
});
renderSymbols();

(function renderErrors() {
  if (!P.errors.length) return;
  document.getElementById("errors-section").innerHTML =
    `<h2 style="font-size:15px;margin:0 0 10px;">Errors (${P.errors.length})</h2>` +
    P.errors.map(e => `<div class="error-card"><b>${esc(e.symbol)}</b>${e.name ? " " + esc(e.name) : ""}: ${esc(e.error_msg || "unknown error")}</div>`).join("");
})();

(function renderHeader() {
  const items = [
    ["As of", P.as_of],
    ["Fitted", P.summary.n_ok],
    ["Errors", P.summary.n_errors],
    ["Changed today", P.summary.n_changed_today],
    ["Revised", P.summary.n_revised],
  ];
  document.getElementById("meta-grid").innerHTML = items.map(([k, v]) =>
    `<div class="meta-item"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");
})();

(function renderFooter() {
  document.getElementById("footer").innerHTML = `
    <span><b>Produced by</b> ${esc(ROW.produced_by)}</span>
    <span><b>Cadence</b> ${esc(ROW.cadence)}</span>
    <span><b>Source</b> ${esc(P.provenance.source)}</span>
    <span><b>Periods/year</b> ${esc(P.provenance.periods_per_year)}</span>
    <span><b>Saved</b> ${esc(ROW.created_at)}</span>
  `;
})();
</script>
"""


def render_html(row: dict[str, Any]) -> str:
    """Render one saved daily run as a self-contained HTML page.

    Args:
        row: The dict :meth:`lazystats.io.depot.ResultDepot.load` returns, whose
            ``payload`` is what
            :func:`~lazystats.regimes.daily_payload.build_payload` wrote.

    Returns:
        A complete page. Pure — no I/O of any kind.
    """
    # `</` cannot appear raw inside a <script> block: a symbol or an error
    # message carrying one would end the script early and blank the page.
    return _TEMPLATE.replace("__ROW_JSON__", json.dumps(row).replace("</", "<\\/"))
