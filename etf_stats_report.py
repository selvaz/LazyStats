"""Render an ``etf_daily_stats`` depot row as a self-contained HTML report.

``render_html(row)`` is a pure function of the exact dict shape
``lazystats.io.depot.ResultDepot.load()`` returns -- no live database
access, no network calls. Every value the report needs (including each
instrument's asset-class/area classification) is embedded in the row's
own ``payload`` at save time (see ``run_daily_etf_stats.fetch``), so any
saved row -- today's or any past day's -- can always be re-rendered from
its JSON alone. See ``render_etf_report.py`` for a CLI that does exactly
that from a stored ``result_id``.
"""

from __future__ import annotations

import json

__all__ = ["render_html"]

_TEMPLATE = r"""<title>ETF Daily Stats — depot artifact</title>
<style>
  :root {
    --bg: #F5F6FA;
    --surface: #FFFFFF;
    --surface-2: #ECEEF3;
    --ink: #12151C;
    --ink-soft: #4B5468;
    --ink-faint: #8890A0;
    --border: #DDE1E8;
    --accent: #B8842A;
    --accent-ink: #6B4E17;
    --pos: #B24A2E;
    --neg: #2E6E8E;
    --eq: #4C6FA6;
    --fi: #4E8A6E;
    --cm: #B8842A;
    --alt: #8B5FA6;
    --fx: #3F8E96;
    --re: #A15C45;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(20, 24, 34, 0.04), 0 8px 24px -12px rgba(20, 24, 34, 0.12);
  }
  :root[data-theme="dark"] {
    --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
    --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
    --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
    --pos: #D97B5C; --neg: #5DA0C4;
    --eq: #7C9BD0; --fi: #7DBA9B; --cm: #D9A54A; --alt: #B490D4; --fx: #63B8C0; --re: #C88870;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0D0F15; --surface: #151822; --surface-2: #1D212D;
      --ink: #E9EAF0; --ink-soft: #9AA3B7; --ink-faint: #656E82;
      --border: #262B38; --accent: #D9A54A; --accent-ink: #F3D9A0;
      --pos: #D97B5C; --neg: #5DA0C4;
      --eq: #7C9BD0; --fi: #7DBA9B; --cm: #D9A54A; --alt: #B490D4; --fx: #63B8C0; --re: #C88870;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased; }
  body { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }
  .mono { font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-variant-numeric: tabular-nums; }

  header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 16px; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
  .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: var(--accent-ink); margin: 0 0 6px; }
  h1 { font-size: 26px; font-weight: 700; letter-spacing: -.01em; margin: 0; text-wrap: balance; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, max-content));
    gap: 4px 28px; text-align: right; }
  .meta-item .k { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); display: block; }
  .meta-item .v { font-size: 13px; color: var(--ink-soft); }

  section { margin-bottom: 40px; }
  .section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
    margin-bottom: 14px; flex-wrap: wrap; }
  h2 { font-size: 15px; font-weight: 700; letter-spacing: .01em; margin: 0; }
  .section-note { font-size: 12.5px; color: var(--ink-faint); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); }

  .chip { display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px; font-weight: 600;
    letter-spacing: .02em; padding: 2px 7px 2px 6px; border-radius: 999px; white-space: nowrap; }
  .chip .dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
  .chip.EQUITY       { color: var(--eq); background: color-mix(in srgb, var(--eq) 14%, transparent); }
  .chip.FIXED_INCOME { color: var(--fi); background: color-mix(in srgb, var(--fi) 14%, transparent); }
  .chip.COMMODITIES  { color: var(--cm); background: color-mix(in srgb, var(--cm) 16%, transparent); }
  .chip.ALTERNATIVES { color: var(--alt); background: color-mix(in srgb, var(--alt) 14%, transparent); }
  .chip.FX           { color: var(--fx); background: color-mix(in srgb, var(--fx) 14%, transparent); }
  .chip.REAL_ESTATE  { color: var(--re); background: color-mix(in srgb, var(--re) 14%, transparent); }
  .chip.UNKNOWN      { color: var(--ink-faint); background: var(--surface-2); }
  .chip .dot.EQUITY       { background: var(--eq); }
  .chip .dot.FIXED_INCOME { background: var(--fi); }
  .chip .dot.COMMODITIES  { background: var(--cm); }
  .chip .dot.ALTERNATIVES { background: var(--alt); }
  .chip .dot.FX           { background: var(--fx); }
  .chip .dot.REAL_ESTATE  { background: var(--re); }
  .chip .dot.UNKNOWN      { background: var(--ink-faint); }

  /* ---------- notable shifts ---------- */
  .shift-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 720px) { .shift-cols { grid-template-columns: 1fr; } }
  .shift-list { padding: 6px 18px 14px; }
  .shift-row { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid var(--border); }
  .shift-row:last-child { border-bottom: none; }
  .shift-label { font-size: 12.5px; display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
  .shift-label b { font-size: 13px; }
  .shift-values { font-size: 11.5px; color: var(--ink-faint); text-align: right; }
  .shift-delta { font-family: ui-monospace, monospace; font-weight: 700; font-size: 13.5px; text-align: right; min-width: 62px; }
  .shift-delta.up { color: var(--pos); }
  .shift-delta.down { color: var(--neg); }
  .subhead { font-size: 12px; font-weight: 700; color: var(--ink-soft); padding: 12px 18px 2px; text-transform: uppercase; letter-spacing: .04em; }

  /* ---------- ticker + fund name (reused in every table except the heatmap) ---------- */
  .tblock { display: flex; flex-direction: column; gap: 2px; overflow: hidden; }
  .tline { display: flex; align-items: center; gap: 6px; }
  .tline b { font-weight: 700; font-size: 13px; }
  .tname { font-size: 10.5px; color: var(--ink-faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }

  /* ---------- volatility rows ---------- */
  .vol-table { padding: 6px 20px 16px; }
  .vol-row { display: grid; grid-template-columns: 168px 1fr 96px; align-items: center; gap: 14px;
    padding: 7px 0; border-bottom: 1px solid var(--border); }
  .vol-row:last-child { border-bottom: none; }
  .bars { position: relative; height: 20px; }
  .bar-track { position: absolute; inset: 0; border-radius: 4px; background: var(--surface-2); }
  .bar { position: absolute; top: 3px; height: 6px; border-radius: 3px; }
  .bar.long  { background: var(--ink-faint); opacity: .55; top: 3px; }
  .bar.short { background: var(--accent); top: 11px; }
  .vol-value { font-size: 12px; text-align: right; }
  .vol-value .short { color: var(--ink); font-weight: 600; }
  .vol-value .sep { color: var(--ink-faint); margin: 0 3px; }
  .vol-value .long { color: var(--ink-faint); }
  .legend { display: flex; gap: 16px; align-items: center; font-size: 11.5px; color: var(--ink-faint); }
  .legend-swatch { display: inline-flex; align-items: center; gap: 5px; }
  .legend-swatch span.sw { width: 12px; height: 4px; border-radius: 2px; }

  /* ---------- returns table ---------- */
  .returns-wrap { overflow-x: auto; padding: 4px 20px 18px; }
  table.returns { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.returns th { text-align: right; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 700; padding: 10px 8px; border-bottom: 1px solid var(--border); }
  table.returns th:first-child, table.returns td:first-child { text-align: left; }
  table.returns td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  table.returns tr:last-child td { border-bottom: none; }
  table.returns td.ticker-cell { text-align: left; }
  .ret-cell { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; line-height: 1.25; }
  .ret-pct { font-weight: 700; }
  .ret-multiple { font-size: 10px; opacity: .8; }

  /* ---------- heatmap ---------- */
  .toggle-group { display: inline-flex; background: var(--surface-2); border-radius: 999px; padding: 3px; gap: 2px; }
  .toggle-group button { border: none; background: transparent; font: inherit; font-size: 12px; font-weight: 600;
    color: var(--ink-soft); padding: 6px 14px; border-radius: 999px; cursor: pointer; }
  .toggle-group button.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }
  .heatmap-wrap { overflow-x: auto; padding: 18px 20px 22px; }
  #heatmap { display: grid; border-collapse: collapse; width: max-content; font-size: 10.5px; }
  .hm-corner, .hm-colhead, .hm-rowhead, .hm-cell { display: flex; align-items: center; justify-content: center; width: 30px; height: 26px; }
  .hm-colhead { writing-mode: vertical-rl; transform: rotate(180deg); font-weight: 700; color: var(--ink-soft);
    height: 60px; align-items: flex-end; padding-bottom: 4px; }
  .hm-rowhead { justify-content: flex-end; padding-right: 8px; font-weight: 700; color: var(--ink-soft); width: 52px; }
  .hm-cell { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums;
    color: var(--ink); font-size: 9px; cursor: default; border-radius: 2px; }
  .hm-cell:hover { outline: 1.5px solid var(--ink); outline-offset: -1.5px; z-index: 2; }
  .hm-tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 11.5px;
    padding: 6px 9px; border-radius: 6px; box-shadow: var(--shadow); z-index: 50; display: none; white-space: nowrap; }

  /* ---------- outliers table ---------- */
  .outliers-wrap { overflow-x: auto; padding: 4px 20px 18px; }
  table.outliers { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.outliers th { text-align: right; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 700; padding: 10px 8px; border-bottom: 1px solid var(--border); }
  table.outliers th:first-child, table.outliers th:nth-child(2), table.outliers th:nth-child(3),
  table.outliers td:first-child, table.outliers td:nth-child(2), table.outliers td:nth-child(3) { text-align: left; }
  table.outliers td { text-align: right; padding: 8px; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-variant-numeric: tabular-nums; }
  table.outliers tr:last-child td { border-bottom: none; }
  table.outliers td.ticker-cell { font-family: -apple-system, "Segoe UI", sans-serif; font-weight: 700; }
  td.dir-positive { color: var(--pos); }
  td.dir-negative { color: var(--neg); }
  .dir-pill { display: inline-block; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 11px;
    font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  .dir-pill.positive { color: var(--pos); background: color-mix(in srgb, var(--pos) 14%, transparent); }
  .dir-pill.negative { color: var(--neg); background: color-mix(in srgb, var(--neg) 14%, transparent); }
  .outliers-empty { padding: 24px 20px; text-align: center; color: var(--ink-faint); font-size: 13px; }

  /* ---------- outlier frequency chart ---------- */
  .freq-wrap { padding: 20px 20px 10px; }
  .freq-chart-area { position: relative; }
  .freq-chart { display: flex; align-items: stretch; gap: 3px; height: 170px; overflow-x: auto; position: relative; z-index: 1; }
  .freq-col { display: flex; flex-direction: column; flex: 1 0 16px; min-width: 16px; height: 100%; }
  .freq-top, .freq-bottom { flex: 1; display: flex; justify-content: center; }
  .freq-top { align-items: flex-end; }
  .freq-bottom { align-items: flex-start; }
  .freq-bar { width: 45%; min-width: 4px; min-height: 0; }
  .freq-bar.pos { background: var(--pos); border-radius: 2px 2px 0 0; }
  .freq-bar.neg { background: var(--neg); border-radius: 0 0 2px 2px; }
  /* The real zero-axis: an overlay pinned to the vertical midpoint of
     .freq-chart (the actual boundary between the positive/negative
     halves), not a decorative line placed below the whole chart. */
  .freq-zero-line { position: absolute; left: 0; right: 0; top: 50%; height: 1.5px;
    background: var(--ink-soft); z-index: 0; pointer-events: none; }
  .freq-labels { display: flex; gap: 3px; margin-top: 6px; }
  .freq-label { flex: 1 0 16px; min-width: 16px; font-size: 8.5px; color: var(--ink-faint); text-align: center;
    writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; }
  .freq-legend { display: flex; gap: 16px; align-items: center; font-size: 11.5px; color: var(--ink-faint); }

  footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 11.5px;
    color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 22px; }
  footer b { color: var(--ink-soft); font-weight: 600; }
</style>

<header>
  <div>
    <p class="eyebrow">lazystats_depot &middot; scheduled artifact</p>
    <h1>ETF Daily Stats</h1>
  </div>
  <div class="meta-grid mono" id="meta-grid"></div>
</header>

<section>
  <div class="section-head">
    <h2>Notable shifts &mdash; short vs. long window</h2>
    <span class="section-note">Biggest movers between the two windows, not just their absolute levels.</span>
  </div>
  <div class="shift-cols">
    <div class="card">
      <p class="subhead">Volatility</p>
      <div class="shift-list" id="vol-shifts"></div>
    </div>
    <div class="card">
      <p class="subhead">Correlation</p>
      <div class="shift-list" id="corr-shifts"></div>
    </div>
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Annualized volatility &mdash; short vs. long window</h2>
    <div class="legend">
      <span class="legend-swatch"><span class="sw" style="background:var(--accent)"></span>short</span>
      <span class="legend-swatch"><span class="sw" style="background:var(--ink-faint);opacity:.55"></span>long</span>
    </div>
  </div>
  <div class="card vol-table" id="vol-table"></div>
</section>

<section>
  <div class="section-head">
    <h2>Trailing returns</h2>
    <span class="section-note" id="returns-note"></span>
  </div>
  <div class="card returns-wrap">
    <table class="returns" id="returns-table"></table>
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Correlation matrix</h2>
    <div class="toggle-group" id="hm-toggle">
      <button data-w="short" class="active">Short window</button>
      <button data-w="long">Long window</button>
    </div>
  </div>
  <div class="card heatmap-wrap"><div id="heatmap"></div></div>
  <div class="section-note" style="margin-top:10px" id="hm-note"></div>
</section>

<section>
  <div class="section-head">
    <h2>Return outliers &mdash; last week</h2>
    <span class="section-note" id="outlier-note"></span>
  </div>
  <div class="card outliers-wrap">
    <table class="outliers" id="outlier-table"></table>
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Outlier frequency &mdash; last month</h2>
    <div class="freq-legend">
      <span class="legend-swatch"><span class="sw" style="background:var(--pos)"></span>positive</span>
      <span class="legend-swatch"><span class="sw" style="background:var(--neg)"></span>negative</span>
    </div>
  </div>
  <div class="card freq-wrap">
    <div class="freq-chart-area">
      <div class="freq-chart" id="freq-chart"></div>
      <div class="freq-zero-line"></div>
    </div>
    <div class="freq-labels" id="freq-labels"></div>
  </div>
</section>

<footer id="footer"></footer>
<div class="hm-tip" id="hm-tip"></div>

<script>
const ROW = __ROW_JSON__;
const P = ROW.payload;
const META = P.instrument_meta;

function stripPrefix(k) { return k.includes(":") ? k.split(":")[1] : k; }

const DATA = {
  result_id: ROW.result_id,
  produced_by: ROW.produced_by,
  series_key: ROW.series_key,
  cadence: ROW.cadence,
  created_at: ROW.created_at,
  as_of: P.as_of,
  instruments: META,
  volatility_short_weeks: P.volatility_short.window_weeks,
  volatility_long_weeks: P.volatility_long.window_weeks,
  volatility: Object.fromEntries(META.map(m => [m.ticker, {
    short: P.volatility_short.volatility[`ticker:${m.ticker}`]?.annualized_volatility ?? null,
    long: P.volatility_long.volatility[`ticker:${m.ticker}`]?.annualized_volatility ?? null,
  }])),
  correlation_short: Object.fromEntries(META.map(m => [m.ticker,
    Object.fromEntries(META.map(n => [n.ticker, P.correlation_short.correlation[`ticker:${m.ticker}`]?.[`ticker:${n.ticker}`] ?? null]))])),
  correlation_long: Object.fromEntries(META.map(m => [m.ticker,
    Object.fromEntries(META.map(n => [n.ticker, P.correlation_long.correlation[`ticker:${m.ticker}`]?.[`ticker:${n.ticker}`] ?? null]))])),
  outliers: P.outliers_last5.outliers,
  outliers_window: P.outliers_last5.window_trading_days,
  outlier_daily_counts: P.outlier_daily_counts,
  returns_table: Object.fromEntries(META.map(m => [m.ticker, P.returns_table[`ticker:${m.ticker}`] || {}])),
  provenance: ROW.provenance,
};
const clsByTicker = Object.fromEntries(META.map(m => [m.ticker, m.asset_class]));
const nameByTicker = Object.fromEntries(META.map(m => [m.ticker, m.name || m.ticker]));

function fmtPct(v) { return v == null ? "&mdash;" : (v * 100).toFixed(1) + "%"; }
function chip(ticker) {
  const cls = clsByTicker[ticker] || "UNKNOWN";
  return `<span class="chip ${cls}"><span class="dot ${cls}"></span>${cls.replace("_"," ")}</span>`;
}
// Ticker + fund name, stacked -- used everywhere EXCEPT the correlation
// heatmap, which stays ticker-only (labels there are already tight at
// 22x22).
function tickerBlock(ticker, extra) {
  return `<span class="tblock"><span class="tline"><b>${ticker}</b>${extra || ""}</span><span class="tname">${nameByTicker[ticker] || ""}</span></span>`;
}

// ---------- header ----------
(function renderHeader() {
  const items = [
    ["As of", DATA.as_of],
    ["Instruments", DATA.instruments.length],
    ["Series key", DATA.series_key],
    ["Result ID", DATA.result_id],
  ];
  document.getElementById("meta-grid").innerHTML = items.map(([k, v]) =>
    `<div class="meta-item"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
})();

// ---------- notable shifts ----------
(function renderShifts() {
  const volShifts = DATA.instruments.map(inst => {
    const v = DATA.volatility[inst.ticker];
    return { ticker: inst.ticker, asset_class: inst.asset_class, delta: (v.short ?? 0) - (v.long ?? 0), short: v.short, long: v.long };
  }).filter(r => r.short != null && r.long != null)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 8);

  document.getElementById("vol-shifts").innerHTML = volShifts.map(r => {
    const dir = r.delta >= 0 ? "up" : "down";
    const arrow = r.delta >= 0 ? "&#9650;" : "&#9660;";
    return `
      <div class="shift-row">
        <span class="shift-label">${tickerBlock(r.ticker, chip(r.ticker))}</span>
        <span class="shift-values mono">${fmtPct(r.short)} / ${fmtPct(r.long)}</span>
        <span class="shift-delta ${dir} mono">${arrow} ${fmtPct(Math.abs(r.delta))}</span>
      </div>`;
  }).join("") || `<div class="shift-row"><span class="shift-label">No data.</span></div>`;

  const tickers = DATA.instruments.map(i => i.ticker);
  const pairShifts = [];
  for (let i = 0; i < tickers.length; i++) {
    for (let j = i + 1; j < tickers.length; j++) {
      const a = tickers[i], b = tickers[j];
      const cs = DATA.correlation_short[a]?.[b], cl = DATA.correlation_long[a]?.[b];
      if (cs == null || cl == null) continue;
      pairShifts.push({ a, b, delta: cs - cl, cs, cl });
    }
  }
  pairShifts.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  document.getElementById("corr-shifts").innerHTML = pairShifts.slice(0, 8).map(r => {
    const dir = r.delta >= 0 ? "up" : "down";
    const arrow = r.delta >= 0 ? "&#9650;" : "&#9660;";
    return `
      <div class="shift-row">
        <span class="shift-label"><b>${r.a}</b> <span class="tname">${nameByTicker[r.a] || ""}</span>&nbsp;&harr;&nbsp;<b>${r.b}</b> <span class="tname">${nameByTicker[r.b] || ""}</span></span>
        <span class="shift-values mono">${r.cs.toFixed(2)} / ${r.cl.toFixed(2)}</span>
        <span class="shift-delta ${dir} mono">${arrow} ${Math.abs(r.delta).toFixed(2)}</span>
      </div>`;
  }).join("");
})();

// ---------- volatility ----------
(function renderVol() {
  const rows = DATA.instruments.map(inst => ({ ...inst, ...DATA.volatility[inst.ticker] }))
    .sort((a, b) => (b.short ?? 0) - (a.short ?? 0));
  const maxVol = Math.max(...rows.map(r => Math.max(r.short ?? 0, r.long ?? 0)));

  document.getElementById("vol-table").innerHTML = rows.map(r => {
    const shortPct = Math.min(100, ((r.short ?? 0) / maxVol) * 100);
    const longPct = Math.min(100, ((r.long ?? 0) / maxVol) * 100);
    return `
      <div class="vol-row">
        ${tickerBlock(r.ticker, chip(r.ticker))}
        <span class="bars">
          <span class="bar-track"></span>
          <span class="bar long" style="width:${longPct}%"></span>
          <span class="bar short" style="width:${shortPct}%"></span>
        </span>
        <span class="vol-value mono"><span class="short">${fmtPct(r.short)}</span><span class="sep">/</span><span class="long">${fmtPct(r.long)}</span></span>
      </div>`;
  }).join("");
})();

// A return of |multiple| >= CAP sigma reaches full color saturation.
const MULTIPLE_COLOR_CAP = 3;
function multipleColor(m) {
  if (m == null) return "transparent";
  const t = Math.max(-MULTIPLE_COLOR_CAP, Math.min(MULTIPLE_COLOR_CAP, m)) / MULTIPLE_COLOR_CAP;
  const mix = Math.round(Math.abs(t) * 55);
  return t >= 0 ? `color-mix(in srgb, var(--pos) ${mix}%, var(--surface))` : `color-mix(in srgb, var(--neg) ${mix}%, var(--surface))`;
}

// ---------- returns table ----------
(function renderReturns() {
  const horizons = DATA.provenance.return_horizons || ["1W", "1M", "3M", "6M", "YTD", "1Y"];
  document.getElementById("returns-note").textContent =
    `Cumulative return per horizon, as of ${DATA.as_of}. Small figure = the return as a multiple of 1Y-annualized volatility scaled to that horizon (σ).`;

  const rows = DATA.instruments.slice().sort((a, b) =>
    a.asset_class.localeCompare(b.asset_class) || a.ticker.localeCompare(b.ticker));

  const thead = `<tr><th>Ticker</th><th>Class</th>${horizons.map(h => `<th>${h}</th>`).join("")}</tr>`;
  const tbody = rows.map(inst => {
    const r = DATA.returns_table[inst.ticker] || {};
    const cells = horizons.map(h => {
      const cell = r[h] || {};
      const v = cell.return;
      const m = cell.vol_multiple;
      if (v == null) return `<td>&mdash;</td>`;
      const sign = v >= 0 ? "+" : "";
      const bg = multipleColor(m);
      const fg = (m != null && Math.abs(m) > 1.8) ? "#fff" : (v >= 0 ? "var(--pos)" : "var(--neg)");
      const mLabel = m == null ? "" : `<span class="ret-multiple">${m >= 0 ? "+" : ""}${m.toFixed(1)}&sigma;</span>`;
      return `<td style="background:${bg}"><span class="ret-cell"><span class="ret-pct" style="color:${fg}">${sign}${(v * 100).toFixed(1)}%</span>${mLabel}</span></td>`;
    }).join("");
    return `<tr><td class="ticker-cell">${tickerBlock(inst.ticker)}</td><td>${chip(inst.ticker)}</td>${cells}</tr>`;
  }).join("");
  document.getElementById("returns-table").innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
})();

// ---------- heatmap ----------
function corrColor(v) {
  if (v == null) return "var(--surface-2)";
  const t = Math.max(-1, Math.min(1, v));
  const mix = Math.round(Math.abs(t) * 62);
  return t >= 0 ? `color-mix(in srgb, var(--pos) ${mix}%, var(--surface))` : `color-mix(in srgb, var(--neg) ${mix}%, var(--surface))`;
}
function corrTextColor(v) { return (v != null && Math.abs(v) > 0.55) ? "#fff" : "var(--ink)"; }

let currentWindow = "short";
function renderHeatmap() {
  const tickers = DATA.instruments.map(i => i.ticker);
  const matrix = currentWindow === "short" ? DATA.correlation_short : DATA.correlation_long;
  const n = tickers.length;
  const grid = document.getElementById("heatmap");
  grid.style.gridTemplateColumns = `52px repeat(${n}, 30px)`;

  let html = `<div class="hm-corner"></div>`;
  tickers.forEach(t => html += `<div class="hm-colhead">${t}</div>`);
  tickers.forEach(rowT => {
    html += `<div class="hm-rowhead">${rowT}</div>`;
    tickers.forEach(colT => {
      const v = matrix[rowT] ? matrix[rowT][colT] : null;
      const bg = corrColor(v);
      const fg = corrTextColor(v);
      const label = v == null ? "" : (rowT === colT ? "1" : v.toFixed(2).replace(/^0\./, ".").replace(/^-0\./, "-."));
      html += `<div class="hm-cell" style="background:${bg};color:${fg}" data-row="${rowT}" data-col="${colT}" data-v="${v}">${label}</div>`;
    });
  });
  grid.innerHTML = html;

  document.getElementById("hm-note").textContent = currentWindow === "short"
    ? `Last ${DATA.volatility_short_weeks} weeks of weekly returns.`
    : `Last ${DATA.volatility_long_weeks} weeks of weekly returns.`;

  const tip = document.getElementById("hm-tip");
  grid.querySelectorAll(".hm-cell").forEach(cell => {
    cell.addEventListener("mouseenter", () => {
      const v = cell.dataset.v;
      if (v === "null" || v === "") return;
      tip.textContent = `${cell.dataset.row} <-> ${cell.dataset.col}: ${parseFloat(v).toFixed(3)}`;
      tip.style.display = "block";
    });
    cell.addEventListener("mousemove", e => { tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px"; });
    cell.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}
document.getElementById("hm-toggle").addEventListener("click", e => {
  const btn = e.target.closest("button");
  if (!btn) return;
  currentWindow = btn.dataset.w;
  document.querySelectorAll("#hm-toggle button").forEach(b => b.classList.toggle("active", b === btn));
  renderHeatmap();
});
renderHeatmap();

// ---------- outliers table (last week) ----------
(function renderOutliers() {
  document.getElementById("outlier-note").textContent =
    `${DATA.outliers.length} event(s) across ${DATA.outliers_window[0]} -> ${DATA.outliers_window[DATA.outliers_window.length - 1]}`;

  const table = document.getElementById("outlier-table");
  if (!DATA.outliers.length) {
    table.innerHTML = "";
    table.parentElement.innerHTML = `<div class="outliers-empty">No outliers in the last week.</div>`;
    return;
  }

  const sorted = DATA.outliers.slice().sort((a, b) => Math.abs(b.z_score) - Math.abs(a.z_score));
  const thead = `<tr><th>Date</th><th>Ticker</th><th>Class</th><th>Direction</th><th>Return</th><th>Z-score</th></tr>`;
  const tbody = sorted.map(o => {
    const ticker = stripPrefix(o.instrument);
    // Plain ASCII "-" from the number's own sign (matches the trailing-returns
    // table) -- a hand-picked unicode minus sign renders as mojibake in some
    // viewers/fonts, ASCII "-" never does.
    const retPct = o.log_return >= 0 ? "+" : "";
    const zPct = o.z_score >= 0 ? "+" : "";
    return `
      <tr>
        <td>${o.date}</td>
        <td class="ticker-cell">${tickerBlock(ticker)}</td>
        <td>${chip(ticker)}</td>
        <td><span class="dir-pill ${o.direction}">${o.direction}</span></td>
        <td class="dir-${o.direction}">${retPct}${(o.log_return * 100).toFixed(2)}%</td>
        <td class="dir-${o.direction}">${zPct}${o.z_score.toFixed(2)}</td>
      </tr>`;
  }).join("");
  table.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
})();

// ---------- outlier frequency chart (last month) ----------
(function renderFrequencyChart() {
  const days = DATA.outlier_daily_counts.window_trading_days;
  const counts = DATA.outlier_daily_counts.counts;
  const maxCount = Math.max(1, ...days.map(d => Math.max(counts[d].positive, counts[d].negative)));

  const chart = document.getElementById("freq-chart");
  chart.innerHTML = days.map(d => {
    const posPct = (counts[d].positive / maxCount) * 100;
    const negPct = (counts[d].negative / maxCount) * 100;
    const posTitle = `${d}: ${counts[d].positive} positive`;
    const negTitle = `${d}: ${counts[d].negative} negative`;
    return `
      <div class="freq-col">
        <div class="freq-top">${counts[d].positive ? `<div class="freq-bar pos" style="height:${posPct}%" title="${posTitle}"></div>` : ""}</div>
        <div class="freq-bottom">${counts[d].negative ? `<div class="freq-bar neg" style="height:${negPct}%" title="${negTitle}"></div>` : ""}</div>
      </div>`;
  }).join("");

  document.getElementById("freq-labels").innerHTML = days.map(d => `<span class="freq-label">${d.slice(5)}</span>`).join("");
})();

// ---------- footer ----------
(function renderFooter() {
  const p = DATA.provenance;
  document.getElementById("footer").innerHTML = `
    <span><b>Produced by</b> ${DATA.produced_by}</span>
    <span><b>Cadence</b> ${DATA.cadence}</span>
    <span><b>Windows</b> ${p.short_window_weeks}w short / ${p.long_window_weeks}w long / ${p.one_year_window_weeks}w vol baseline</span>
    <span><b>Outlier threshold</b> |z| &ge; ${p.outlier_threshold}</span>
    <span><b>Source</b> ${p.source}</span>
    <span><b>Saved</b> ${DATA.created_at}</span>
  `;
})();
</script>
"""


def render_html(row: dict) -> str:
    """Render ``row`` (the dict shape :meth:`ResultDepot.load` returns) as a
    self-contained HTML report. Pure function -- no I/O -- so it works
    identically whether ``row`` just came off a live pipeline run or was
    loaded from the depot days later."""
    return _TEMPLATE.replace("__ROW_JSON__", json.dumps(row))
