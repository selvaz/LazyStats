"""Render an ``anomaly_explanation`` depot row as a self-contained HTML report.

``render_html(row)`` is a pure function of the exact dict shape
``lazystats.io.depot.ResultDepot.load()`` returns -- same contract as
``etf_stats_report.render_html`` (and same CSS variable palette, for visual
consistency across LazyStats reports), applied to narrative findings
instead of a numeric dashboard.
"""

from __future__ import annotations

import json

__all__ = ["render_html"]

_TEMPLATE = r"""<title>Anomaly Explanations — depot artifact</title>
<style>
  :root {
    --bg: #F5F6FA; --surface: #FFFFFF; --surface-2: #ECEEF3;
    --ink: #12151C; --ink-soft: #4B5468; --ink-faint: #8890A0;
    --border: #DDE1E8; --accent: #B8842A; --accent-ink: #6B4E17;
    --pos: #B24A2E; --neg: #2E6E8E;
    --cat-monetary_policy: #4C6FA6; --cat-macro_data: #4E8A6E;
    --cat-geopolitical: #B8842A; --cat-company_specific: #8B5FA6;
    --cat-liquidity_technical: #3F8E96; --cat-unclear: #8890A0;
    --conf-high: #2E6E8E; --conf-medium: #B8842A; --conf-low: #8890A0;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(20, 24, 34, 0.04), 0 8px 24px -12px rgba(20, 24, 34, 0.12);
  }
  :root[data-theme="dark"] {
    --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
    --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
    --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
    --pos: #D97B5C; --neg: #5DA0C4;
    --cat-monetary_policy: #7C9BD0; --cat-macro_data: #7DBA9B;
    --cat-geopolitical: #D9A54A; --cat-company_specific: #B490D4;
    --cat-liquidity_technical: #63B8C0; --cat-unclear: #656E82;
    --conf-high: #5DA0C4; --conf-medium: #D9A54A; --conf-low: #656E82;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
      --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
      --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
      --pos: #D97B5C; --neg: #5DA0C4;
      --cat-monetary_policy: #7C9BD0; --cat-macro_data: #7DBA9B;
      --cat-geopolitical: #D9A54A; --cat-company_specific: #B490D4;
      --cat-liquidity_technical: #63B8C0; --cat-unclear: #656E82;
      --conf-high: #5DA0C4; --conf-medium: #D9A54A; --conf-low: #656E82;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased; }
  body { max-width: 920px; margin: 0 auto; padding: 28px 24px 64px; }
  .mono { font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
    font-variant-numeric: tabular-nums; }

  header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 16px; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
  .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: var(--accent-ink); margin: 0 0 6px; }
  h1 { font-size: 24px; font-weight: 700; letter-spacing: -.01em; margin: 0; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, max-content));
    gap: 4px 24px; text-align: right; }
  .meta-item .k { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); display: block; }
  .meta-item .v { font-size: 13px; color: var(--ink-soft); }

  .finding { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 18px 20px; margin-bottom: 16px; }
  .finding-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .finding-title { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .finding-title b { font-size: 15px; }
  .finding-type { font-size: 11.5px; color: var(--ink-faint); }
  .badges { display: flex; gap: 6px; align-items: center; }
  .chip { display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px; font-weight: 600;
    letter-spacing: .02em; padding: 3px 8px; border-radius: 999px; white-space: nowrap; }
  .chip .dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
  .conf-pill { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    text-transform: uppercase; letter-spacing: .03em; }

  .explanation { font-size: 13.5px; line-height: 1.55; color: var(--ink); margin: 0 0 10px; }
  .detail-row { font-size: 11.5px; color: var(--ink-faint); margin-bottom: 10px; }
  .detail-row span { margin-right: 14px; }
  .evidence-list { padding: 0; margin: 0; list-style: none; border-top: 1px solid var(--border); padding-top: 10px; }
  .evidence-item { font-size: 12px; color: var(--ink-soft); margin-bottom: 6px; padding-left: 14px; position: relative; }
  .evidence-item::before { content: "\2014"; position: absolute; left: 0; color: var(--ink-faint); }
  .evidence-item b { color: var(--ink); }
  .evidence-empty { font-size: 12px; color: var(--ink-faint); font-style: italic; }

  footer { margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 11.5px;
    color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 22px; }
  footer b { color: var(--ink-soft); font-weight: 600; }
</style>

<header>
  <div>
    <p class="eyebrow">anomaly_explanations &middot; depot artifact</p>
    <h1 id="title">Anomaly Explanations</h1>
  </div>
  <div class="meta-grid mono" id="meta-grid"></div>
</header>

<div id="findings"></div>
<footer id="footer"></footer>

<script>
const ROW = __ROW_JSON__;
const P = ROW.payload;

function chip(category) {
  const label = category.replace(/_/g, " ");
  return `<span class="chip" style="color:var(--cat-${category});background:color-mix(in srgb, var(--cat-${category}) 16%, transparent)">
    <span class="dot" style="background:var(--cat-${category})"></span>${label}</span>`;
}
function confPill(conf) {
  return `<span class="conf-pill" style="color:var(--surface);background:var(--conf-${conf})">${conf}</span>`;
}
function fmtDetail(d) {
  return Object.entries(d).map(([k, v]) => {
    const num = typeof v === "number" ? (Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(3)) : v;
    return `<span><b>${k}</b>: ${num}</span>`;
  }).join("");
}

document.getElementById("title").textContent = `Anomaly Explanations — ${P.date}`;
document.getElementById("meta-grid").innerHTML = [
  ["Date", P.date],
  ["Items", P.items.length],
  ["Trigger", P.trigger_result_id],
  ["Result ID", ROW.result_id],
].map(([k, v]) => `<div class="meta-item"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");

document.getElementById("findings").innerHTML = P.items.map(item => {
  const evidence = (item.evidence && item.evidence.length)
    ? `<ul class="evidence-list">${item.evidence.map(e =>
        `<li class="evidence-item"><b>${e.source}</b>${e.date ? " (" + e.date + ")" : ""} — ${e.detail}</li>`).join("")}</ul>`
    : `<div class="evidence-list evidence-empty">No grounded evidence found.</div>`;
  return `
    <div class="finding">
      <div class="finding-head">
        <div class="finding-title"><b>${item.instrument}</b><span class="finding-type">${item.anomaly_type.replace(/_/g, " ")} · ${item.date}</span></div>
        <div class="badges">${chip(item.category)}${confPill(item.confidence)}</div>
      </div>
      <p class="explanation">${item.explanation}</p>
      <div class="detail-row mono">${fmtDetail(item.detail || {})}</div>
      ${evidence}
    </div>`;
}).join("");

document.getElementById("footer").innerHTML = `
  <span><b>Produced by</b> ${ROW.produced_by}</span>
  <span><b>Cadence</b> ${ROW.cadence}</span>
  <span><b>Saved</b> ${ROW.created_at}</span>
`;
</script>
"""


def render_html(row: dict) -> str:
    """Render ``row`` (the dict shape :meth:`ResultDepot.load` returns) as a
    self-contained HTML report. Pure function -- no I/O."""
    return _TEMPLATE.replace("__ROW_JSON__", json.dumps(row))
