"""Render a ``weekly_anomaly_review`` depot row as a self-contained HTML
report. Same contract/palette as ``anomaly_report.render_html`` and
``etf_stats_report.render_html``.
"""

from __future__ import annotations

import json

__all__ = ["render_html"]

_TEMPLATE = r"""<title>Weekly Anomaly Review — depot artifact</title>
<style>
  :root {
    --bg: #F5F6FA; --surface: #FFFFFF; --surface-2: #ECEEF3;
    --ink: #12151C; --ink-soft: #4B5468; --ink-faint: #8890A0;
    --border: #DDE1E8; --accent: #B8842A; --accent-ink: #6B4E17;
    --pos: #B24A2E; --neg: #2E6E8E;
    --verdict-confirmed: #4E8A6E; --verdict-questionable: #B24A2E; --verdict-unverifiable: #8890A0;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(20, 24, 34, 0.04), 0 8px 24px -12px rgba(20, 24, 34, 0.12);
  }
  :root[data-theme="dark"] {
    --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
    --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
    --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
    --pos: #D97B5C; --neg: #5DA0C4;
    --verdict-confirmed: #7DBA9B; --verdict-questionable: #D97B5C; --verdict-unverifiable: #656E82;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
      --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
      --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
      --pos: #D97B5C; --neg: #5DA0C4;
      --verdict-confirmed: #7DBA9B; --verdict-questionable: #D97B5C; --verdict-unverifiable: #656E82;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased; }
  body { max-width: 920px; margin: 0 auto; padding: 28px 24px 64px; }
  .mono { font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace; }

  header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 16px; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
  .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: var(--accent-ink); margin: 0 0 6px; }
  h1 { font-size: 24px; font-weight: 700; letter-spacing: -.01em; margin: 0; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, max-content));
    gap: 4px 24px; text-align: right; }
  .meta-item .k { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); display: block; }
  .meta-item .v { font-size: 13px; color: var(--ink-soft); }

  section { margin-bottom: 32px; }
  h2 { font-size: 15px; font-weight: 700; margin: 0 0 12px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 18px 20px; }
  .narrative { font-size: 13.5px; line-height: 1.6; margin: 0; }

  .synth-group { margin-top: 14px; }
  .synth-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
    color: var(--ink-faint); margin: 0 0 6px; }
  .synth-list { margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.55; }
  .synth-empty { font-size: 12.5px; color: var(--ink-faint); font-style: italic; }

  table.verify { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.verify th { text-align: left; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 700; padding: 8px; border-bottom: 1px solid var(--border); }
  table.verify td { text-align: left; padding: 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  table.verify tr:last-child td { border-bottom: none; }
  .verdict-pill { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    text-transform: uppercase; white-space: nowrap; color: var(--surface); }

  footer { margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 11.5px;
    color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 22px; }
  footer b { color: var(--ink-soft); font-weight: 600; }
</style>

<header>
  <div>
    <p class="eyebrow">weekly_anomaly_review &middot; depot artifact</p>
    <h1 id="title">Weekly Anomaly Review</h1>
  </div>
  <div class="meta-grid mono" id="meta-grid"></div>
</header>

<section>
  <h2>Synthesis</h2>
  <div class="card" id="synthesis"></div>
</section>

<section>
  <h2>Verifications</h2>
  <div class="card" style="padding:0; overflow-x:auto;"><table class="verify" id="verify-table"></table></div>
</section>

<footer id="footer"></footer>

<script>
const ROW = __ROW_JSON__;
const P = ROW.payload;

function verdictPill(v) {
  return `<span class="verdict-pill" style="background:var(--verdict-${v})">${v}</span>`;
}
function synthList(items) {
  if (!items || !items.length) return `<p class="synth-empty">None flagged.</p>`;
  return `<ul class="synth-list">${items.map(i => `<li>${i}</li>`).join("")}</ul>`;
}

document.getElementById("title").textContent = `Weekly Anomaly Review — ${P.week_start} → ${P.week_end}`;
document.getElementById("meta-grid").innerHTML = [
  ["Week", `${P.week_start} → ${P.week_end}`],
  ["Daily items", P.n_daily_items],
  ["Result ID", ROW.result_id],
].map(([k, v]) => `<div class="meta-item"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");

document.getElementById("synthesis").innerHTML = `
  <p class="narrative">${P.synthesis.narrative}</p>
  <div class="synth-group"><p class="synth-label">New trends</p>${synthList(P.synthesis.new_trends)}</div>
  <div class="synth-group"><p class="synth-label">Regime confirmations</p>${synthList(P.synthesis.regime_confirmations)}</div>
  <div class="synth-group"><p class="synth-label">New risks</p>${synthList(P.synthesis.new_risks)}</div>
`;

const thead = `<tr><th>Instrument</th><th>Type</th><th>Date</th><th>Verdict</th><th>Note</th></tr>`;
const tbody = P.verifications.map(v => `
  <tr>
    <td><b>${v.instrument}</b></td>
    <td>${v.anomaly_type.replace(/_/g, " ")}</td>
    <td class="mono">${v.date}</td>
    <td>${verdictPill(v.verdict)}</td>
    <td>${v.note}</td>
  </tr>`).join("");
document.getElementById("verify-table").innerHTML = P.verifications.length
  ? `<thead>${thead}</thead><tbody>${tbody}</tbody>`
  : `<tbody><tr><td style="color:var(--ink-faint);font-style:italic;">No daily items this week.</td></tr></tbody>`;

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
